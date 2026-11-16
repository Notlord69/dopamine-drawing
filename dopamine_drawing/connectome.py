"""The MaleCNS v1.0 signed weight set and tools to carve subnetworks out of it.

A subnetwork is a dict(keep, pre, post, w): `keep` holds the global neuron indices (sorted),
`pre`/`post` are local indices 0..len(keep)-1, and `w` is the signed synapse count of each
connection (sign from the presynaptic neuron's predicted transmitter). lif.TorchBatchLIF runs one.

Carried over from fly-connectome-arm (sim/lif.py).
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp


# ---------------------------------------------------------------- weight set
class WeightSet:
    """Loads weights_male-cns_v1.0/ (made by scripts/fetch_connectome.py)."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / 'meta.json').read_text())
        self.neurons = pd.read_parquet(self.folder / 'neurons_signs.parquet')
        e = pq.read_table(self.folder / 'edges.parquet')
        self.pre = e['pre_idx'].to_numpy()
        self.post = e['post_idx'].to_numpy()
        self.count = e['syn_count'].to_numpy()
        self.N = len(self.neurons)
        self._out = None   # rows = pre, cols = post, synapse counts

    def signed(self, sign_col='sign'):
        """Signed synapse count for every edge ('sign' or 'sign_modexc')."""
        return self.count * self.neurons[sign_col].to_numpy()[self.pre].astype(np.int32)

    def idx(self, mask):
        """Neuron indices where a boolean mask over ws.neurons is True."""
        return self.neurons.index[np.asarray(mask)].to_numpy()

    def by_type(self, pattern, cls=None):
        """Neuron indices whose type matches a regex (optionally within a class)."""
        m = self.neurons['type'].astype(str).str.match(pattern)
        if cls is not None:
            m &= self.neurons['cls'] == cls
        return self.idx(m)

    def _adj(self):
        if self._out is None:
            self._out = sp.csr_matrix((self.count, (self.pre, self.post)),
                                      shape=(self.N, self.N))
        return self._out

    def partners(self, idx, direction='out', min_syn=1):
        """Direct partners of a set of neurons, with summed synapse counts."""
        A = self._adj()
        A = A if direction == 'out' else A.T.tocsr()
        sub = A[np.asarray(idx)]
        tot = np.asarray(sub.sum(axis=0)).ravel()
        keep = np.flatnonzero(tot >= min_syn)
        return pd.Series(tot[keep], index=keep, name='syn_from_group')

    def distances(self, seeds, n_hops, direction='out', min_syn=1, verbose=True):
        """Hop distance from seeds for every neuron (-1 = not reached within n_hops).
        Each hop uses only connections with >= min_syn synapses."""
        A = self._adj()
        A = A if direction == 'out' else A.T.tocsr()
        A = A.multiply(A >= min_syn).tocsr()
        dist = np.full(self.N, -1, np.int16)
        dist[seeds] = 0
        frontier = np.asarray(seeds)
        for h in range(1, n_hops + 1):
            nxt = np.unique(A[frontier].indices)
            frontier = nxt[dist[nxt] < 0]
            dist[frontier] = h
            if verbose:
                print(f"  hop {h}: +{len(frontier):,} new, {(dist >= 0).sum():,} total")
        return dist

    def hops(self, seeds, n_hops, direction='out', min_syn=1, verbose=True):
        """Neurons reachable within n_hops (each hop: edges with >= min_syn)."""
        return np.flatnonzero(self.distances(seeds, n_hops, direction, min_syn, verbose) >= 0)

    def path_neurons(self, src, dst, max_len, min_syn=1, verbose=True):
        """Neurons on some path src -> ... -> dst of at most max_len hops:
        dist_from_src + dist_to_dst <= max_len."""
        if verbose: print("downstream of source:")
        df = self.distances(src, max_len, 'out', min_syn, verbose)
        if verbose: print("upstream of target:")
        db = self.distances(dst, max_len, 'in', min_syn, verbose)
        ok = (df >= 0) & (db >= 0) & (df.astype(int) + db <= max_len)
        return np.flatnonzero(ok), df, db

    def subnetwork(self, keep, sign_col='sign'):
        """Edges among `keep` only, renumbered 0..len(keep)-1."""
        keep = np.unique(keep)
        local = np.full(self.N, -1, np.int64)
        local[keep] = np.arange(len(keep))
        m = (local[self.pre] >= 0) & (local[self.post] >= 0)
        w = self.signed(sign_col)[m]
        nz = w != 0
        return dict(keep=keep,
                    pre=local[self.pre[m]][nz].astype(np.int32),
                    post=local[self.post[m]][nz].astype(np.int32),
                    w=w[nz].astype(np.float64))

    def path_subnetwork(self, src, dst, max_len=7, min_syn=5, max_conn=2_500_000,
                        extra=(), verbose=True):
        """Subnetwork of neurons on src -> dst paths of length <= L, choosing the
        largest L whose subnetwork has <= max_conn connections. `extra` neurons
        (e.g. candidates to record) are always included."""
        _, df, db = self.path_neurons(src, dst, max_len, min_syn, verbose)
        ok = (df >= 0) & (db >= 0)
        total = df.astype(int) + db
        reach = df[np.asarray(dst)]
        reach = reach[reach >= 0]
        if len(reach) == 0:
            raise ValueError(f"no target reachable within {max_len} hops")
        shortest = int(reach.min())
        always = np.concatenate([np.asarray(src), np.asarray(dst), np.asarray(extra, dtype=int)])
        if verbose:
            print(f"\nshortest path: {shortest} hops; targets reachable within {max_len}: "
                  f"{(df[np.asarray(dst)] >= 0).sum()}/{len(dst)}\n")
        best, table = None, []
        for L in range(shortest, max_len + 1):
            keep = np.union1d(np.flatnonzero(ok & (total <= L)), always)
            sub = self.subnetwork(keep)
            table.append(dict(L=L, neurons=len(keep), connections=len(sub['w'])))
            if verbose:
                print(f"L = {L}: {len(keep):>7,} neurons, {len(sub['w']):>10,} connections")
            if len(sub['w']) > max_conn:
                break
            best = (L, sub)
        if best is None:
            raise ValueError("even the shortest-path subnetwork exceeds max_conn")
        if verbose:
            print(f"\nusing L = {best[0]}")
        return best[1], best[0], pd.DataFrame(table), df


def shuffle_sub(sub, seed=0):
    """Degree-preserving random rewiring of a subnetwork: targets are permuted
    across all connections, so every neuron keeps its number of outgoing and
    incoming connections, and each connection keeps its weight and sign
    (sign follows the presynaptic neuron)."""
    rng = np.random.default_rng(seed)
    return dict(sub, post=rng.permutation(sub['post']))
