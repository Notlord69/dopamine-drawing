"""Leaky integrate-and-fire model of Shiu et al. 2024 and a batched torch engine (frozen weights).

Neuron and synapse model as in Shiu et al. 2024, model.py (github.com/philshiu/Drosophila_brain_model),
re-implemented with exact exponential integration at dt = 0.1 ms. Weights are signed synapse counts
(sub['w']) times a global synapse strength w_syn * w_scale. Input neurons are driven by Poisson spike
trains through a fixed suprathreshold synapse (w_syn * f_poi), as in Shiu et al.

TorchBatchLIF runs B trials side by side (CUDA, or CPU for testing). Trials that finish can be
dropped from the batch with keep(). Poisson input comes from a counter-based hash of (trial seed,
time step, input neuron), so a trial's input does not depend on which batch it ran in.

Carried over from fly-connectome-arm (sim/batch7.py), where it was checked spike for spike
against Brian2 in float64. On CUDA, synaptic inputs are summed with atomic adds in varying order,
so float32 spikes are not bit-reproducible between runs (rates are).

Weights are frozen here. The plastic (R-STDP) engine is the Week 1-2 work (PLAN.md).
"""
import math, time

import numpy as np

# Shiu et al. model.py default_params, in ms / mV / Hz
SHIU = dict(
    v_0=-52.0,     # mV, resting potential
    v_rst=-52.0,   # mV, reset potential after a spike
    v_th=-45.0,    # mV, spike threshold
    t_mbr=20.0,    # ms, membrane time constant
    tau=5.0,       # ms, synaptic time constant
    t_rfc=2.2,     # ms, refractory period
    t_dly=1.8,     # ms, synaptic delay
    w_syn=0.275,   # mV per synapse (tuned on FlyWire; the parent project used w_scale 0.70 on MaleCNS)
    r_poi=150.0,   # Hz, stimulus rate
    f_poi=250,     # stimulus synapse = w_syn * f_poi (always suprathreshold)
)

M32 = 0xFFFFFFFF
_K = 0x45D9F3B                    # < 2^27, so products of 32-bit values stay inside int64
_GOLD = 0x9E3779B1


def _h(x):
    """32-bit integer hash (works on numpy int64 arrays/scalars and torch int64 tensors)."""
    x = (((x >> 16) ^ x) * _K) & M32
    x = (((x >> 16) ^ x) * _K) & M32
    return (x >> 16) ^ x


def _seed_base(seeds, n_stim):
    """(B, n_stim) int64: one hash per (episode seed, input neuron)."""
    sb = _h(np.asarray(seeds, np.int64) & M32)
    j = (((np.arange(n_stim, dtype=np.int64) + 1) * _GOLD) & M32)
    return _h(sb[:, None] ^ j[None, :])


def _step_hash(t):
    """Hash of the global step numbers t (int64 array)."""
    return _h((np.asarray(t, np.int64) * 0x85EBCA6B + 0x27D4EB2F) & M32)


def poisson_input_np(base, t0, n_steps, prob):
    """bool (n_steps, B, n_stim): stimulus spikes for steps t0..t0+n_steps-1.
    base (B, n_stim) from _seed_base, prob (B, n_stim) = rate * dt."""
    ht = _step_hash(np.arange(t0, t0 + n_steps))
    u = _h(base[None, :, :] ^ ht[:, None, None])
    return u < np.round(np.asarray(prob) * 2.0 ** 32).astype(np.int64)[None]


def _consts(p, dt_ms):
    """Exact per-step update constants for the LIF equations (exponential integration)."""
    c = {k: float(p[k]) for k in ('v_0', 'v_rst', 'v_th', 'w_syn', 'r_poi')}
    t_mbr, tau = float(p['t_mbr']), float(p['tau'])
    c['a'] = math.exp(-dt_ms / t_mbr)
    c['c'] = math.exp(-dt_ms / tau)
    c['b'] = tau / (tau - t_mbr) * (c['c'] - c['a'])
    c['delay'] = int(round(float(p['t_dly']) / dt_ms))
    c['rfc'] = int((float(p['t_rfc']) + 1e-3 * dt_ms) / dt_ms)
    return c


# ---------------------------------------------------------------- torch (GPU)
class TorchBatchLIF:
    """Batched LIF engine in torch. State is (B, n); B changes as trials are dropped (keep())."""
    kind = 'torch'

    def __init__(self, sub, stim_global, device='cuda', dtype='float32', params=None, dt_ms=0.1,
                 verbose=True):
        import torch
        self.torch = torch
        p = dict(SHIU, **(params or {}))
        self.dev = torch.device(device)
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.keep_ids = np.asarray(sub['keep'])
        n = self.n = len(self.keep_ids)
        self.stim_np = np.searchsorted(self.keep_ids, np.asarray(stim_global))
        assert np.array_equal(self.keep_ids[self.stim_np], np.asarray(stim_global)), "inputs not in network"
        self.dt = dt_ms
        c = _consts(p, dt_ms)
        for k in ('v_0', 'v_rst', 'v_th', 'w_syn', 'r_poi', 'a', 'b', 'c', 'delay'):
            setattr(self, k, c[k])
        self.w_stim = SHIU['w_syn'] * SHIU['f_poi']       # fixed: does not follow w_scale
        rfc = np.full(n, c['rfc'], np.int32); rfc[self.stim_np] = 0
        t0 = time.perf_counter()
        kw = dict(device=self.dev)
        self.rfc_steps = torch.tensor(rfc, **kw)
        self.stim = torch.tensor(self.stim_np, dtype=torch.long, **kw)
        pre, post, w = np.asarray(sub['pre']), np.asarray(sub['post']), np.asarray(sub['w'])
        order = np.argsort(pre, kind='stable')
        indptr = np.zeros(n + 1, np.int64)
        np.cumsum(np.bincount(pre, minlength=n), out=indptr[1:])
        self.indptr = torch.tensor(indptr, **kw)
        self.post = torch.tensor(post[order].astype(np.int64), **kw)
        self.c_w = torch.tensor(w[order], dtype=self.dtype, **kw)
        self.n_conn = len(w)
        del order
        self._sync()
        self.build_s = time.perf_counter() - t0
        if verbose:
            print(f"torch on {self.dev} ({str(self.dtype).split('.')[-1]}): {n:,} neurons, "
                  f"{self.n_conn:,} connections, built in {self.build_s:.1f} s")

    def _sync(self):
        if self.dev.type == 'cuda':
            self.torch.cuda.synchronize(self.dev)

    def reset(self, seeds, w_scales, track=False):
        torch, n = self.torch, self.n
        B = self.B = len(seeds)
        kw = dict(device=self.dev)
        self.v = torch.full((B, n), self.v_0, dtype=self.dtype, **kw)
        self.g = torch.zeros((B, n), dtype=self.dtype, **kw)
        self.last = torch.full((B, n), -10 ** 9, dtype=torch.int32, **kw)
        self.ring = [torch.zeros(0, dtype=torch.long, **kw) for _ in range(self.delay)]
        self.step_i = 0
        self.w_eff = torch.tensor(np.asarray(w_scales, float) * self.w_syn, dtype=self.dtype, **kw)
        self.base = torch.tensor(_seed_base(seeds, len(self.stim_np)), **kw)
        self.track = track
        self.tot = torch.zeros((B, n), dtype=torch.int32, **kw) if track else None

    def keep(self, idx):
        """Keep only trials idx (positions in the current batch, ascending)."""
        torch, n = self.torch, self.n
        idx_t = torch.as_tensor(np.asarray(idx, np.int64), device=self.dev)
        remap = torch.full((self.B,), -1, dtype=torch.long, device=self.dev)
        remap[idx_t] = torch.arange(len(idx), device=self.dev)
        for k in ('v', 'g', 'last', 'w_eff', 'base') + (('tot',) if self.track else ()):
            setattr(self, k, getattr(self, k)[idx_t])
        for s, fl in enumerate(self.ring):
            if len(fl):
                nb = remap[fl // n]
                m = nb >= 0
                self.ring[s] = nb[m] * n + fl[m] % n
        self.B = len(idx)

    def advance(self, prob, n_steps, rec_local, bin_steps):
        """prob: (B, n_stim) spike probability per step. Returns spike counts of the neurons
        rec_local (local indices) in bins of bin_steps steps: numpy uint16 (B, n_bins, n_rec)."""
        torch, n, B = self.torch, self.n, self.B
        dev = self.dev
        rec = torch.as_tensor(np.asarray(rec_local, np.int64), device=dev)
        thr = torch.as_tensor(np.round(np.asarray(prob) * 2.0 ** 32).astype(np.int64), device=dev)[None]
        buf = torch.zeros((n_steps, B, len(rec)), dtype=torch.bool, device=dev)
        CH = 400                     # stimulus spikes drawn 40 ms at a time (few kernels, bounded memory)
        for k in range(n_steps):
            if k % CH == 0:
                ht = torch.as_tensor(_step_hash(np.arange(self.step_i, self.step_i + min(CH, n_steps - k))),
                                     device=dev)
                inp = _h(self.base[None] ^ ht[:, None, None]) < thr                # (<=CH, B, n_stim)
            t = self.step_i
            nr = (t - self.last) >= self.rfc_steps
            v_new = self.v_0 + (self.v - self.v_0) * self.a + self.g * self.b
            self.v = torch.where(nr, v_new, self.v)
            self.g = torch.where(nr, self.g * self.c, self.g)
            spk = (self.v > self.v_th) & nr
            flat = spk.view(-1).nonzero().squeeze(1)
            self.last.view(-1)[flat] = t
            open_ = (nr & ~spk).to(self.dtype)
            old = self.ring[t % self.delay]
            if len(old):
                i = old % n
                starts = self.indptr[i]
                cnt = self.indptr[i + 1] - starts
                total = int(cnt.sum())
                if total:
                    rep = torch.repeat_interleave(torch.arange(len(i), device=dev), cnt, output_size=total)
                    first = torch.cumsum(cnt, 0) - cnt
                    e = starts[rep] + torch.arange(total, device=dev) - first[rep]
                    bb = (old // n)[rep]
                    tgt = self.post[e] + bb * n
                    self.g.view(-1).index_add_(0, tgt, self.c_w[e] * open_.view(-1)[tgt] * self.w_eff[bb])
            self.ring[t % self.delay] = flat
            self.v[:, self.stim] += inp[k % CH].to(self.dtype) * open_[:, self.stim] * self.w_stim
            self.v.view(-1)[flat] = self.v_rst
            self.g.view(-1)[flat] = 0
            buf[k] = spk[:, rec]
            if self.track:
                self.tot += spk
            self.step_i += 1
        nb = n_steps // bin_steps
        counts = buf.view(nb, bin_steps, B, len(rec)).sum(1, dtype=torch.int32).permute(1, 0, 2)
        return counts.to(torch.int32).cpu().numpy().astype(np.uint16)

    def totals(self):
        return self.tot.cpu().numpy()
