#!/usr/bin/env python
"""Download the FlyEM male CNS v1.0 connectome and build the files the code expects.

    python scripts/fetch_connectome.py [--data-dir DIR] [--neurons FILE] [--skip-shortlist]

Writes into the data folder (default: $FLYARM_DATA_DIR or <repo>/data):

    raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather   official bulk export (1.1 GB, download)
    neurons_male-cns_v1.0.parquet       neuron table from neuPrint (male-cns:v1.0)
    shortlist_week1.csv                 335 JO-C/E inputs, 68 DN candidates, 81 front-leg MNs
    dn_ranking_week1.csv                every DN's input/output path score (shortlist's source)
    connections_male-cns_v1.0.parquet   neuron-to-neuron connections (176,422 neurons)
    weights_male-cns_v1.0/              neurons_signs.parquet, edges.parquet, W_signed_counts.npz, meta.json
    wsyn_choice.json                    w_scale 0.70 (week 2's choice, copied from results/week2/)
    week7/week7_random_network_meta.json  the week 7 random network's w_scale (copied from results/week7/)

This is the project's week 1-2 pipeline as one script (archive/notebooks/01_status.ipynb and
02_weights_VERSION2.ipynb), same queries, rules and outputs. The neuron table and the DN shortlist
come from neuPrint, which needs a (free) token: https://neuprint.janelia.org -> Account -> Auth token,
then `export NEUPRINT_TOKEN=...`. Never put the token in a file in this repo.

Data: FlyEM male CNS connectome v1.0 (HHMI Janelia FlyEM, University of Cambridge, MRC LMB,
Google Research), CC-BY 4.0. See ATTRIBUTION.md.

Memory: the filter runs in pyarrow without a pandas copy (it ran in a 4 GB WSL session).
"""
import argparse, json, os, shutil, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RELEASE = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
F_WEIGHTS_NAME = 'connectome-weights-male-cns-v1.0-minconf-0.5.feather'
NEUPRINT = 'https://neuprint.janelia.org'
DATASET = 'male-cns:v1.0'
EXPECT = dict(neurons=176_422, connections=25_862_574, baseline_edges=24_287_791,
              jo=335, dn_candidates=68, leg_mn=81)


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


# ---------------------------------------------------------------- week 1: neuPrint tables
def neuprint_client():
    tok = os.environ.get('NEUPRINT_TOKEN')
    if not tok:
        sys.exit("NEUPRINT_TOKEN is not set. Get a token at https://neuprint.janelia.org (Account -> Auth token), "
                 "then: export NEUPRINT_TOKEN=...   (or pass --neurons with an existing neuron table)")
    import neuprint as neu
    return neu.Client(NEUPRINT, dataset=DATASET, token=tok)


def fetch_neurons(c, out):
    """01_status.ipynb cells 7-9: every Neuron with type, class, transmitter and side."""
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


def fetch_shortlist(c, neurons, data):
    """01_status.ipynb cells 2-4 and 6: JO-C/E inputs, front-leg (ProLN) motor neurons, and the DN
    candidates ranked by two-hop input and output path strength (connections with >= 5 synapses)."""
    from neuprint import fetch_adjacencies, NeuronCriteria as NC
    extra = c.fetch_custom("""
        MATCH (n:Neuron)
        WHERE n.superclass IN ['vnc_motor', 'cb_sensory']
        RETURN n.bodyId AS bodyId, n.somaNeuromere AS neuromere,
               n.exitNerve AS exitNerve, n.entryNerve AS entryNerve
    """)
    df = neurons.merge(extra, on='bodyId', how='left')
    mn_all = df[df.cls == 'vnc_motor']
    S = df[(df.cls == 'cb_sensory') & df['type'].astype(str).str.match(r'^JO-(C|E)')]
    MN = mn_all[mn_all['neuromere'].astype(str).str.contains('T1') & (mn_all['exitNerve'] == 'ProLN')]
    DN = df[df.cls == 'descending_neuron']
    say(f"  JO-C/E {len(S)}, front-leg (ProLN) MNs {len(MN)}, DNs {len(DN)}")

    post_total = neurons.set_index('bodyId')['post']

    def edges(src=None, tgt=None):
        """Connections with >=5 synapses, as a fraction of the target's total input."""
        _, conn = fetch_adjacencies(NC(bodyId=list(src)) if src is not None else None,
                                    NC(bodyId=list(tgt)) if tgt is not None else None,
                                    min_total_weight=5)
        e = conn.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)['weight'].sum()
        e['frac'] = e['weight'] / e['bodyId_post'].map(post_total)
        return e

    S_ids, DN_ids, MN_ids = set(S.bodyId), set(DN.bodyId), set(MN.bodyId)
    e1 = edges(src=S_ids)
    e2 = edges(src=set(e1.bodyId_post), tgt=DN_ids)
    two_in = e1.merge(e2, left_on='bodyId_post', right_on='bodyId_pre', suffixes=('_1', '_2'))
    two_in['path'] = two_in.frac_1 * two_in.frac_2
    in_score = (two_in.groupby('bodyId_post_2')['path'].sum()
                .add(e1[e1.bodyId_post.isin(DN_ids)].groupby('bodyId_post')['frac'].sum(), fill_value=0))
    e3 = edges(tgt=MN_ids)
    e4 = edges(src=DN_ids, tgt=set(e3.bodyId_pre))
    two_out = e4.merge(e3, left_on='bodyId_post', right_on='bodyId_pre', suffixes=('_1', '_2'))
    two_out['path'] = two_out.frac_1 * two_out.frac_2
    out_score = (two_out.groupby('bodyId_pre_1')['path'].sum()
                 .add(e3[e3.bodyId_pre.isin(DN_ids)].groupby('bodyId_pre')['frac'].sum(), fill_value=0))
    rank = pd.DataFrame({'in_score': in_score, 'out_score': out_score}).fillna(0)
    rank['score'] = np.sqrt(rank.in_score * rank.out_score)
    rank = rank.join(DN.set_index('bodyId')[['type', 'instance']]).sort_values('score', ascending=False)
    rank.to_csv(data / 'dn_ranking_week1.csv')

    pick = (set(rank.nlargest(10, 'in_score').index) | set(rank.nlargest(10, 'out_score').index)
            | set(rank.nlargest(10, 'score').index))
    types = rank.loc[list(pick), 'type'].unique()          # both L and R of each type
    DN_mid = DN[DN['type'].isin(types)].merge(rank[['in_score', 'out_score', 'score']],
                                              left_on='bodyId', right_index=True)
    DN_mid['role'] = np.where(DN_mid.in_score > 3 * DN_mid.out_score, 'input-side',
                              np.where(DN_mid.out_score > 3 * DN_mid.in_score, 'output-side', 'bridge'))
    cols = ['bodyId', 'type', 'instance', 'side', 'nt']
    shortlist = pd.concat([S[cols].assign(group='input_JO-CE'),
                           DN_mid[cols + ['in_score', 'out_score', 'score', 'role']].assign(group='middle_descending'),
                           MN[cols].assign(group='output_T1_leg_MN')], ignore_index=True)
    shortlist.to_csv(data / 'shortlist_week1.csv', index=False)
    counts = shortlist['group'].value_counts().to_dict()
    say(f"  shortlist: {counts}")
    for g, k in (('input_JO-CE', 'jo'), ('middle_descending', 'dn_candidates'), ('output_T1_leg_MN', 'leg_mn')):
        if counts.get(g) != EXPECT[k]:
            say(f"  WARNING: {g} has {counts.get(g)} neurons; the project used {EXPECT[k]}")
    return shortlist


# ---------------------------------------------------------------- week 2: connection table and weights
def build_connections(f_weights, neurons, out):
    """02_weights_VERSION2.ipynb cell B: neuron-to-neuron rows of the bulk export, no synapse cutoff."""
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
    """02_weights_VERSION2.ipynb cells C-G: sign table, edge list in model indices, signed matrix, meta."""
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
            say(f"  WARNING: {k} = {v:,}; the project's weight set has {EXPECT[k]:,}")
    return meta


def copy_params(data):
    """The two small run parameters the notebooks read, from results/."""
    pairs = [(REPO / 'results' / 'week2' / 'wsyn_choice.json', data / 'wsyn_choice.json'),
             (REPO / 'results' / 'week7' / 'week7_random_network_meta.json',
              data / 'week7' / 'week7_random_network_meta.json')]
    for src, dst in pairs:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy(src, dst)
        say(f"  {dst.relative_to(data)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', type=Path,
                    default=Path(os.environ.get('FLYARM_DATA_DIR', REPO / 'data')))
    ap.add_argument('--neurons', type=Path, help='use this neuron table instead of querying neuPrint')
    ap.add_argument('--weights-feather', type=Path, help='use this local copy of the bulk weights file')
    ap.add_argument('--skip-shortlist', action='store_true', help="don't query the DN shortlist (needs neuPrint)")
    ap.add_argument('--delete-raw', action='store_true', help='delete the 1.1 GB download when done')
    a = ap.parse_args()
    data = a.data_dir.resolve()
    data.mkdir(parents=True, exist_ok=True)
    say(f"data folder: {data}")

    say("1. neuron table (neuPrint)")
    f_neurons = data / 'neurons_male-cns_v1.0.parquet'
    c = None
    if a.neurons:
        neurons = pd.read_parquet(a.neurons)
        if a.neurons.resolve() != f_neurons:
            neurons.to_parquet(f_neurons)
    elif f_neurons.exists():
        neurons = pd.read_parquet(f_neurons)
        say(f"  have {f_neurons.name}")
    else:
        c = neuprint_client()
        neurons = fetch_neurons(c, f_neurons)

    say("2. DN shortlist (neuPrint)")
    if a.skip_shortlist:
        say("  skipped")
    elif (data / 'shortlist_week1.csv').exists():
        say("  have shortlist_week1.csv")
    else:
        fetch_shortlist(c or neuprint_client(), neurons, data)

    say("3. bulk connectome export")
    f_w = a.weights_feather or download(RELEASE + F_WEIGHTS_NAME, data / 'raw' / F_WEIGHTS_NAME)

    say("4. neuron-to-neuron connections")
    tbl = build_connections(f_w, neurons, data / 'connections_male-cns_v1.0.parquet')

    say("5. weight set")
    build_weights(tbl, neurons, data / 'weights_male-cns_v1.0', Path(f_w).name)
    del tbl

    say("6. run parameters")
    copy_params(data)
    if a.delete_raw and not a.weights_feather:
        (data / 'raw' / F_WEIGHTS_NAME).unlink(missing_ok=True)
    say("done")


if __name__ == '__main__':
    main()
