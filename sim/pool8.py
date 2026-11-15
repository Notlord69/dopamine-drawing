"""Week 8b: where is the information lost? Input, inside the brain, or at the readout?

Week 8 (fix B, real connectome): 0/50. The video shows one failure mode everywhere: near the target
the decoded commands shrink toward zero (ridge regression to the mean), so the arm stops where they
cancel. It hovers holding the cube, or releases and never lifts away (the retreat). The expert's
rules weren't the limit; the information in the readout was. Week 7's open-loop decoding agrees:
expert dx R2 0.27 and dz 0.39 from all descending neurons.

This module records, while the expert drives (identical states for every readout):
  * the 335 JO-C/E input neurons (their spikes = what the encoder delivers; the ceiling),
  * all descending neurons (week 7/8's readout),
  * every other neuron that fires >= 1 Hz during an expert episode ('active' central neurons),
and compares, out of episode (5 folds by whole episode, ridge, penalty chosen inside each fold):
  expert dx, dy, dz R2; grip; the gain of the decoded motion near the target (1 = unbiased, near 0 =
  shrunk to zero, the hover); dz sign agreement while lowering and while retreating; the 8 signals.

  active_neurons      neurons firing >= min_hz during a whole expert episode (one brain run)
  collect_expert      expert-driven episodes, recording any set of neurons; saved as an npz
  pool_report         the comparison table
  gate                the rule (fixed before the run) for whether to run the closed loop on a wider pool
"""
import time
from pathlib import Path

import numpy as np

from .pickplace import SIGNALS8, ACT4, informative, dagger_pp, EnvPool, normalise8
from .readout import ridge_fit_predict, r2, _folds


# ---------------------------------------------------------------- which neurons fire
def active_neurons(brain, w_scale, states, candidates_local, min_hz=1.0):
    """Local indices (from candidates_local) of neurons firing >= min_hz while the brain is driven
    through `states` (one 40 ms step each, after week 5's 200 ms warm-up). Returns (active, hz)."""
    from .week7 import frac_firing_batch
    _, hz = frac_firing_batch(brain, [w_scale], states, candidates_local)
    hz = np.asarray(hz)[0]
    cand = np.asarray(candidates_local)
    return cand[hz[cand] >= min_hz], hz


# ---------------------------------------------------------------- expert-driven recording
def collect_expert(brain, w_scale, obs_names, env, seeds, ex, out_path, brain_ms=80, window_ms=80,
                   noise=0.005, max_batch=64, resume=True):
    """The expert drives `seeds` (every other one with `noise` m of action noise per step, as DAgger
    round 0) while the brain watches; rates of every neuron in brain.record over the last window_ms,
    at every step. Saved to out_path (npz: X float32 rows x len(record), Y expert labels, g episode,
    obs). Nothing is fitted."""
    out_path = Path(out_path)
    if resume and out_path.exists():
        d = np.load(out_path)
        print(f"  loaded {out_path.name}: {d['X'].shape[0]:,} steps x {d['X'].shape[1]:,} neurons")
        return {k: d[k] for k in d.files}
    from .week8 import Week8BrainPolicy
    pol = Week8BrainPolicy(brain, w_scale, obs_names, pool=np.arange(len(brain.record)), k=None,
                           brain_ms=brain_ms, window_ms=window_ms, name='expert drives, brain watches',
                           max_batch=max_batch)
    pol.fit = lambda *a, **k: None                 # recording only
    t0 = time.perf_counter()
    data, _ = dagger_pp(EnvPool(env), pol, ex, [list(seeds)], [1.0], noise=noise, verbose=False)
    d = dict(X=data['X'].astype(np.float32), Y=data['Y'], g=data['g'], obs=data['obs'])
    np.savez(out_path, **d)
    print(f"  recorded {len(d['Y']):,} steps x {d['X'].shape[1]:,} neurons from {len(seeds)} expert episodes "
          f"in {(time.perf_counter() - t0) / 60:.1f} min")
    return d


# ---------------------------------------------------------------- the comparison
def _cv_pred(X, Y, g, k=None, n_folds=5, seed=0):
    """Out-of-episode ridge predictions of Y; with k, the k most informative columns are chosen
    inside each training fold (no leakage)."""
    P = np.zeros_like(Y, dtype=float)
    for f in _folds(g, n_folds, seed):
        cols = np.arange(X.shape[1]) if k is None or k >= X.shape[1] else informative(X[~f], Y[~f], k)[0]
        P[f], _ = ridge_fit_predict(X[~f][:, cols], Y[~f], X[f][:, cols], g[~f])
    return P


def pool_report(d, pools, ex, obs_names, near_mm=50, dead_mm=3.0):
    """d: from collect_expert. pools: {name: column indices into d['X']} or {name: ('signals',)} for the
    noise-free 8 signals, or {name: (cols, k)} to keep the k most informative (chosen in-fold).
    Returns a list of rows (one per pool)."""
    X, Y, g, obs = d['X'], d['Y'].astype(float), d['g'], d['obs']
    phase = np.array([ex(o)[1] for o in obs])
    S = normalise8(obs, obs_names)
    near = (np.hypot(obs[:, 19], obs[:, 20]) < near_mm / 1000) & (obs[:, 10] > 0.03) & (phase != 'retreat')
    lower = (phase == 'lower') & (Y[:, 2] < -dead_mm / 1000)
    retreat = (phase == 'retreat') & (Y[:, 2] > dead_mm / 1000)
    rows = []
    for name, spec in pools.items():
        t0 = time.perf_counter()
        if isinstance(spec, tuple) and spec and isinstance(spec[0], str):
            F, k = S, None
        elif isinstance(spec, tuple):
            F, k = X[:, np.asarray(spec[0])].astype(float), spec[1]
        else:
            F, k = X[:, np.asarray(spec)].astype(float), None
        P = _cv_pred(F, Y, g, k)
        r = r2(Y[:, :3], P[:, :3])
        lab = Y[:, 3] > 0
        bal = 0.5 * (np.mean((P[:, 3] > 0)[lab]) + np.mean((P[:, 3] <= 0)[~lab]))
        # gain near the target: slope of decoded on expert motion (x, y pooled). 1 = unbiased.
        yn, pn = Y[near, :2].ravel(), P[near, :2].ravel()
        gain = float(np.cov(pn, yn)[0, 1] / np.var(yn)) if near.sum() > 10 else np.nan
        sig = _cv_pred(F, S, g, k) if not (isinstance(spec, tuple) and spec and isinstance(spec[0], str)) else S
        rs = r2(S, sig)
        rows.append({'readout': name, 'neurons': F.shape[1] if k is None else min(k, F.shape[1]),
                     'dx R2': round(float(r[0]), 3), 'dy R2': round(float(r[1]), 3), 'dz R2': round(float(r[2]), 3),
                     'grip bal. acc.': round(float(bal), 3),
                     'motion gain near target': round(gain, 2),
                     'dz right sign, lowering': round(float(np.mean(P[lower, 2] < 0)), 2) if lower.any() else np.nan,
                     'dz right sign, retreat': round(float(np.mean(P[retreat, 2] > 0)), 2) if retreat.any() else np.nan,
                     'signals R2 (median)': round(float(np.median(rs)), 3),
                     'signals R2 (min)': round(float(np.min(rs)), 3),
                     'weakest signal': SIGNALS8[int(np.argmin(rs))],
                     's': round(time.perf_counter() - t0, 1)})
        print(f"  {name}: done in {rows[-1]['s']} s")
    counts = dict(rows=len(Y), episodes=len(np.unique(g)), near_target=int(near.sum()),
                  lowering=int(lower.sum()), retreat=int(retreat.sum()))
    return rows, counts


def score(row):
    """One number per readout: mean of dx, dy, dz R2 and the two dz sign agreements."""
    return float(np.mean([row['dx R2'], row['dy R2'], row['dz R2'],
                          row['dz right sign, lowering'], row['dz right sign, retreat']]))


def gate(rows, base='all DNs (week 8)', wide='active, 3,000 most informative', ceiling='JO input spikes (ceiling)',
         margin=0.10):
    """Rule fixed before the run:
      * the wider pool beats the DN readout by >= margin on score()  -> run the closed loop on it;
      * otherwise, if the JO input ceiling is no better than the DNs either -> the encoder loses the
        information: revisit the input (more neurons per channel), not the readout;
      * otherwise the information reaches the brain but not the readout neurons we can afford."""
    R = {r['readout']: r for r in rows}
    sb, sw, sc = score(R[base]), score(R[wide]), score(R[ceiling])
    if sw >= sb + margin:
        return 'closed loop', (f"the wider readout carries clearly more ({sw:.2f} vs {sb:.2f}, margin {margin}): "
                               f"run the closed loop with it")
    if sc < sb + margin:
        return 'encoder', (f"even the input spikes score only {sc:.2f} (DNs {sb:.2f}): the encoder is the bottleneck. "
                           f"Next: more input neurons per channel / higher rates, not a wider readout")
    return 'brain', (f"the input carries it ({sc:.2f}) but neither the DNs ({sb:.2f}) nor 3,000 active neurons "
                     f"({sw:.2f}) do: the information is lost inside the network")
