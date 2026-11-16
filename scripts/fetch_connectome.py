#!/usr/bin/env python
"""Download the FlyEM male CNS v1.0 connectome and build the signed weight set.

    python scripts/fetch_connectome.py [--data-dir DIR] [--neurons FILE] [--weights-feather FILE] [--delete-raw]

Writes into the data folder (default: $DD_DATA_DIR or <repo>/data):

    raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather   official bulk export (1.1 GB, download)
    neurons_male-cns_v1.0.parquet       neuron table from neuPrint (male-cns:v1.0)
    connections_male-cns_v1.0.parquet   neuron-to-neuron connections (176,422 neurons)
    weights_male-cns_v1.0/              neurons_signs.parquet, edges.parquet, W_signed_counts.npz, meta.json

If you already have these from fly-connectome-arm, skip this and point DD_DATA_DIR at that folder.

The neuron table comes from neuPrint, which needs a (free) token: https://neuprint.janelia.org ->
Account -> Auth token, then `export NEUPRINT_TOKEN=...`. Never put the token in a file in this repo.

Data: FlyEM male CNS connectome v1.0 (HHMI Janelia FlyEM, University of Cambridge, MRC LMB,
Google Research), CC-BY 4.0. See ATTRIBUTION.md.

Memory: the filter runs in pyarrow without a pandas copy (it ran in a 4 GB WSL session).
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RELEASE = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
F_WEIGHTS_NAME = 'connectome-weights-male-cns-v1.0-minconf-0.5.feather'
NEUPRINT = 'https://neuprint.janelia.org'
DATASET = 'male-cns:v1.0'
EXPECT = dict(neurons=176_422, connections=25_862_574, baseline_edges=24_287_791)


def say(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- download
def download(url, dest, chunk=1 << 22):
    import requests
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        say(f"  have {dest.name} ({dest.stat().st_size / 1e6:,.0f} MB)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        done, t0 = 0, time.time()
        with open(tmp, 'wb') as f:
            for b in r.iter_content(chunk):
                f.write(b)
                done += len(b)
                if total:
                    print(f"\r  {dest.name}: {done / 1e6:,.0f} / {total / 1e6:,.0f} MB "
                          f"({done / max(1e-9, time.time() - t0) / 1e6:.0f} MB/s)", end='', flush=True)
    print()
    if total and tmp.stat().st_size != total:
        raise IOError(f"incomplete download of {url}")
    tmp.rename(dest)
    return dest


# ---------------------------------------------------------------- neuPrint neuron table
def neuprint_client():
    tok = os.environ.get('NEUPRINT_TOKEN')
    if not tok:
        sys.exit("NEUPRINT_TOKEN is not set. Get a token at https://neuprint.janelia.org (Account -> Auth token), "
                 "then: export NEUPRINT_TOKEN=...   (or pass --neurons with an existing neuron table)")
    import neuprint as neu
    return neu.Client(NEUPRINT, dataset=DATASET, token=tok)


def fetch_neurons(c, out):
    """Every Neuron with type, class, transmitter and side."""
    props = c.fetch_custom("""
        MATCH (n:Neuron) WITH n LIMIT 2000
        UNWIND keys(n) AS k
        RETURN k AS property, count(*) AS n ORDER BY n DESC
    """)
    available = set(props['property'])
    pick = lambda cands: next((p for p in cands if p in available), None)
    CLASS = pick(['superclass', 'class'])
    SUBCLASS = pick(['class', 'subclass']) if CLASS == 'superclass' else pick(['subclass'])
    NT = pick(['predictedNt', 'consensusNt', 'celltypePredictedNt', 'predicted_nt'])
    SIDE = pick(['somaSide', 'rootSide', 'side'])
    say(f"  neuPrint fields: class={CLASS} subclass={SUBCLASS} nt={NT} side={SIDE}")
    fields = {'bodyId': 'bodyId', 'type': 'type', 'instance': 'instance', 'pre': 'pre', 'post': 'post',
              'cls': CLASS, 'subcls': SUBCLASS, 'nt': NT, 'side': SIDE}
    ret = ", ".join(f"n.{p} AS {alias}" for alias, p in fields.items() if p)
    neurons = c.fetch_custom(f"MATCH (n:Neuron) RETURN {ret}")
    neurons.to_parquet(out)
    say(f"  {len(neurons):,} neurons -> {out.name}")
    return neurons


# ---------------------------------------------------------------- connection table and weights
def build_connections(f_weights, neurons, out):
    """Neuron-to-neuron rows of the bulk export, no synapse cutoff."""
    import pyarrow as pa, pyarrow.dataset as ds, pyarrow.parquet as pq
    PRE, POST, W = 'body_pre', 'body_post', 'weight'
    dset = ds.dataset(f_weights, format='ipc')
    missing = [c for c in (PRE, POST, W) if c not in dset.schema.names]
    assert not missing, f"columns not found: {missing}; available: {dset.schema.names}"
    if out.exists():
        say(f"  have {out.name}")
        return pq.read_table(out)
    ids = pa.array(neurons['bodyId'].astype('int64').values)
    tbl = dset.to_table(columns=[PRE, POST, W], filter=ds.field(PRE).isin(ids) & ds.field(POST).isin(ids))
    pq.write_table(tbl, out, compression='zstd')
    say(f"  {tbl.num_rows:,} neuron-to-neuron connections -> {out.name}")
    return tbl


def build_weights(tbl, neurons, out_dir, source_name):
    """Sign table, edge list in model indices, signed matrix, meta."""
    import pyarrow as pa, pyarrow.parquet as pq, scipy.sparse as sp
    PRE, POST, W = 'body_pre', 'body_post', 'weight'
    N = len(neurons)
    assert neurons['bodyId'].is_unique, "duplicate bodyIds in neuron table"
    nt_raw = neurons['nt'].astype('string').str.strip().str.lower()
    FAST = {'acetylcholine': +1, 'gaba': -1, 'glutamate': -1, 'histamine': -1}
    MOD = {'dopamine', 'serotonin', 'octopamine', 'tyramine'}
    UNKNOWN = {'unclear', 'none', 'nan', ''}
    unexpected = set(nt_raw.dropna().unique()) - set(FAST) - MOD - UNKNOWN
    if unexpected:
        say(f"  WARNING: unexpected transmitter labels (treated as unknown): {unexpected}")

    known = nt_raw.isin(set(FAST) | MOD)
    typed = neurons['type'].notna() & known
    majority = (pd.DataFrame({'type': neurons['type'][typed], 'nt': nt_raw[typed]})
                .groupby('type')['nt'].agg(lambda s: s.value_counts().idxmax()))
    from_type = neurons['type'].map(majority)
    # histamine is a photoreceptor transmitter: only allowed as a cell-type fill in the optic lobe
    HIST_OK = {'ol_sensory', 'ol_intrinsic'}
    bad_hist = (from_type == 'histamine') & ~neurons['cls'].isin(HIST_OK)
    from_type = from_type.mask(bad_hist)
    nt_used = nt_raw.where(known, from_type).fillna('acetylcholine')
    nt_source = np.select([known, from_type.notna()], ['predicted', 'celltype_majority'], 'default_ach')
    signs = pd.DataFrame({
        'idx': np.arange(N, dtype=np.int32), 'bodyId': neurons['bodyId'].astype('int64').values,
        'type': neurons['type'].values, 'cls': neurons['cls'].values, 'side': neurons['side'].values,
        'nt_raw': nt_raw.values, 'nt_used': nt_used.values, 'nt_source': nt_source, 'guessed': ~known.values})
    signs['sign'] = signs['nt_used'].map(FAST).fillna(0).astype(np.int8)               # modulators -> 0
    signs['sign_modexc'] = np.where(signs['nt_used'].isin(MOD), 1, signs['sign']).astype(np.int8)

    idx_of = pd.Index(signs['bodyId'])
    pre_idx = idx_of.get_indexer(tbl[PRE].to_numpy()).astype(np.int32)
    post_idx = idx_of.get_indexer(tbl[POST].to_numpy()).astype(np.int32)
    assert (pre_idx >= 0).all() and (post_idx >= 0).all(), "connection to a neuron missing from the table"
    count = tbl[W].to_numpy().astype(np.int32)
    sign_pre = signs['sign'].to_numpy()[pre_idx]

    out_dir.mkdir(parents=True, exist_ok=True)
    signs.to_parquet(out_dir / 'neurons_signs.parquet')
    pq.write_table(pa.table({'pre_idx': pre_idx, 'post_idx': post_idx, 'syn_count': count}),
                   out_dir / 'edges.parquet', compression='zstd')
    W_signed = sp.csr_matrix((count * sign_pre, (post_idx, pre_idx)), shape=(N, N), dtype=np.float32)
    W_signed.eliminate_zeros()          # drops modulator edges (sign 0) from this baseline matrix
    sp.save_npz(out_dir / 'W_signed_counts.npz', W_signed)
    meta = {
        'dataset': DATASET, 'source_file': source_name, 'neurons': int(N), 'edges_all': int(len(count)),
        'edges_in_baseline_matrix': int(W_signed.nnz), 'total_synapses': int(count.sum()),
        'synapse_cutoff': None, 'self_connections_kept': True,
        'matrix_layout': 'CSR, rows=post, cols=pre, values = syn_count * sign (no w_syn)',
        'sign_rules': {**FAST, **{m: 0 for m in MOD}},
        'sign_modexc': 'modulators = +1, otherwise as baseline',
        'unknown_fill': 'majority nt of same cell type (known neurons only), else acetylcholine',
        'histamine_fill_only_in': sorted(HIST_OK), 'guessed_neurons': int(signs['guessed'].sum()),
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
    say(f"  weights: {N:,} neurons, {len(count):,} connections, {W_signed.nnz:,} in the signed matrix, "
        f"{int(count.sum()):,} synapses, {meta['guessed_neurons']:,} transmitters guessed")
    for k, v in (('neurons', N), ('connections', len(count)), ('baseline_edges', W_signed.nnz)):
        if v != EXPECT[k]:
            say(f"  WARNING: {k} = {v:,}; the reference weight set has {EXPECT[k]:,}")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', type=Path,
                    default=Path(os.environ.get('DD_DATA_DIR', REPO / 'data')))
    ap.add_argument('--neurons', type=Path, help='use this neuron table instead of querying neuPrint')
    ap.add_argument('--weights-feather', type=Path, help='use this local copy of the bulk weights file')
    ap.add_argument('--delete-raw', action='store_true', help='delete the 1.1 GB download when done')
    a = ap.parse_args()
    data = a.data_dir.resolve()
    data.mkdir(parents=True, exist_ok=True)
    say(f"data folder: {data}")

    say("1. neuron table (neuPrint)")
    f_neurons = data / 'neurons_male-cns_v1.0.parquet'
    if a.neurons:
        neurons = pd.read_parquet(a.neurons)
        if a.neurons.resolve() != f_neurons:
            neurons.to_parquet(f_neurons)
    elif f_neurons.exists():
        neurons = pd.read_parquet(f_neurons)
        say(f"  have {f_neurons.name}")
    else:
        neurons = fetch_neurons(neuprint_client(), f_neurons)

    say("2. bulk connectome export")
    f_w = a.weights_feather or download(RELEASE + F_WEIGHTS_NAME, data / 'raw' / F_WEIGHTS_NAME)

    say("3. neuron-to-neuron connections")
    tbl = build_connections(f_w, neurons, data / 'connections_male-cns_v1.0.parquet')

    say("4. weight set")
    build_weights(tbl, neurons, data / 'weights_male-cns_v1.0', Path(f_w).name)
    del tbl

    if a.delete_raw and not a.weights_feather:
        (data / 'raw' / F_WEIGHTS_NAME).unlink(missing_ok=True)
    say("done")


if __name__ == '__main__':
    main()
