"""Week 10: does the fly's wiring matter, on fresh seeds, and can the last 4 seeds be won?

Week 9: the frozen connectome with readout C4 (3,000 neurons re-chosen on all DAgger data, ridge,
release region up-weighted to 1/3 of the fit) completes pick-and-place on 46/50 test seeds. The
4 failures: 3 hover above the target still holding the cube (the press, the phase C4 traded away),
1 never lifted. Those test seeds (92000-92049) have been looked at in four weeks.

Everything here is tested on FRESH seeds, 93000-93049 (never used before), with week 7-9's training
seeds and DAgger (4 x 30) unchanged:

  1. connectome, C4      week 9's readout, rebuilt from week 9's own DAgger data (refit, no brain
                         time) when week 9's output is attached; otherwise retrained the same way.
  2. random network, C4  week 7's random network (every connection's target shuffled; degrees,
                         weights, E/I kept), its own active neurons, its own 3,000-neuron readout,
                         the same recipe, seeds and timing. Pre-registered comparison: week 7's
                         verdict rule (sim/week7.verdict).
  3. connectome, G       the grip-gated readout (offline in week 9: trap up 0.84, press down 0.90).
                         Grip from one ridge on all rows; motion from one of two ridges, fitted on
                         the rows whose label is open / closed, chosen by the decoded grip. Each map
                         is linear; the switch between them is the one nonlinearity.
  4. connectome, C6      still linear: C4 plus the press rows (holding the cube in the last 4 cm
                         above the table near the target) up-weighted to PRESS_SHARE of the fit.
  5. random network, G   (optional) the comparison repeated with the gated readout.

No-brain controls, trained exactly like the brain (DAgger, same seeds), on the 8 noise-free signals:
expert; MLP; linear (as weeks 7-9); linear with C4's weights; linear with C6's weights; and the gated
readout G. A readout's success counts as the brain's only if its own no-brain control fails.

  share_weights        row weights giving each flagged group (release region, press) a share of the fit
  Week10BrainPolicy    a recipe (C4, C6, G) on the brain; logs the region and press flags as 2 extra
                       columns (outside the readout pool) so fit() can weight rows
  FlagSignalPolicy     the same recipes on the 8 noise-free signals (no brain)
  run_network10        one network and recipe end to end (DAgger or week 9 warm start, fresh-seed test,
                       feedback cut, failure stats, video episodes)
  run_references10     the no-brain controls
  week9_warm           finds week 9's connectome C4 run among the attached inputs
  compare10            the pre-registered verdict and the readout comparisons
"""
import gc, json, pickle, time
from pathlib import Path

import numpy as np

from .expert8 import RELEASE_XY
from .pickplace import (ExpertPolicy, SignalPolicy, dagger_pp, evaluate_pp, EnvPool, normalise8,
                        paired_milestones, LIFT_Z)
from .week7 import light, slug, verdict
from .week8 import Week8BrainPolicy
from .week9 import (K_WIDE, END_SHARE, EXPERTS, WRidge, GatedRidge, informative_w, region_masks,
                    Week9BrainPolicy, trap_stats, _build_brain)

TEST10 = list(range(93_000, 93_050))      # fresh: used in no earlier week
CUT10 = TEST10[:10]
PRESS_SHARE = 1 / 4                       # C6: press rows' share of the fit (natural share ~13%; ~9% under C4)
PASS = 40                                 # of 50, unchanged since week 7
FLAGS = ('region', 'press')

RECIPES = {
    'C4': dict(labels='B', shares={'region': END_SHARE}, gated=False,
               what="week 9: 3,000 neurons re-chosen on all data, release region 1/3 of the fit, ridge"),
    'C6': dict(labels='B', shares={'region': END_SHARE, 'press': PRESS_SHARE}, gated=False,
               what=f"C4 + press rows {PRESS_SHARE:.0%} of the fit (still linear)"),
    'G': dict(labels='B', shares={}, gated=True,
              what="grip-gated: two linear motion maps, switched by the decoded grip (week 9's offline G)"),
}
SIMPLER = ('C4', 'C6', 'G')               # tie-break order for 'best readout'


# ---------------------------------------------------------------- weights and the gated readout
def share_weights(flags, shares):
    """flags: {name: bool mask} (disjoint groups). shares: {name: share of the total weight}.
    A group with fewer rows than its share is weighted up to exactly that share; groups already
    at or above it keep weight 1 and count with the rest. Mean weight 1. None if nothing changes
    (plain fit). With one group this equals week 9's end_weights."""
    n = len(next(iter(flags.values())))
    boost = {k: s for k, s in shares.items() if 0 < flags[k].sum() < s * n}
    if not boost:
        return None
    rest = np.ones(n, bool)
    for k in boost:
        rest &= ~flags[k]
    w = np.zeros(n)
    w[rest] = (1 - sum(boost.values())) / max(1, rest.sum())
    for k, s in boost.items():
        w[flags[k]] = s / flags[k].sum()
    return w * (n / w.sum())


class GatedRidge10(GatedRidge):
    """Week 9's GatedRidge (same fits, same switch), guarded for a side with too few rows to fit
    (only possible in a smoke test): that side then uses the all-rows motion map."""
    MIN_ROWS, MIN_EPISODES = 50, 3

    def fit(self, X, Y, groups, w=None):
        self.all = WRidge().fit(X, Y, groups, w)
        op = Y[:, 3] > 0
        fit_side = lambda m: (WRidge().fit(X[m], Y[m], groups[m], None if w is None else w[m])
                              if m.sum() >= self.MIN_ROWS and len(np.unique(groups[m])) >= self.MIN_EPISODES
                              else self.all)
        self.open, self.closed = fit_side(op), fit_side(~op)
        self.alpha = self.all.alpha
        return self


def make_readout(recipe):
    return GatedRidge10() if recipe['gated'] else WRidge()


def _flags_from(X, start):
    return {k: X[:, start + i] > 0.5 for i, k in enumerate(FLAGS)}


def _raster(cols, lin):
    Wm = np.zeros((len(cols), 4))
    Wm[lin.ok] = lin.W / lin.W.std(0).clip(1e-12)
    top = np.argsort(-np.abs(Wm).max(1))[:100]
    top = top[np.lexsort((-np.abs(Wm[top]).max(1), np.abs(Wm[top]).argmax(1)))]
    return cols[top]


# ---------------------------------------------------------------- policies
class Week10BrainPolicy(Week8BrainPolicy):
    """Week 8b/9's brain policy (80/80 timing, k neurons from the pool) with a week 10 recipe.
    The k neurons are re-chosen at every refit on all data so far, with the recipe's row weights
    (week 9's 'all' selection). act() appends the release-region and press flags of the state
    as two logged columns, outside the pool: the readout never sees them."""

    def __init__(self, *a, recipe=None, **kw):
        super().__init__(*a, **kw)
        self.recipe = RECIPES['C4'] if recipe is None else recipe

    def act(self, obs_list):
        A, X = super().act(obs_list)
        M = region_masks(np.array(obs_list))
        return A, np.c_[X, M['region'].astype(float), M['press'].astype(float)]

    def fit(self, X, Y, g, rounds=None):
        w = share_weights(_flags_from(X, X.shape[1] - 2), self.recipe['shares'])
        if self.k is None or self.k >= len(self.pool):
            sel = np.arange(len(self.pool))
        else:
            sel = informative_w(X[:, self.pool].astype(float), Y, self.k, w)[0]
        self.cols = self.pool[sel]
        self.readout = make_readout(self.recipe).fit(X[:, self.cols].astype(float), Y, g, w)
        self.raster_cols = _raster(self.cols, self.readout.all if self.recipe['gated'] else self.readout)
        return self.readout


class FlagSignalPolicy(SignalPolicy):
    """No brain: a week 10 recipe's readout on the 8 noise-free signals. Logged features are the
    8 signals plus the two flags (for the row weights); the readout sees the 8 signals only."""

    def __init__(self, obs_names, recipe, name):
        super().__init__(obs_names, 'ridge', name=name)
        self.recipe = recipe

    def features(self, obs):
        self.s = normalise8(obs, self.obs_names, self.signals)[0]
        M = region_masks(np.asarray(obs)[None])
        return np.r_[self.s, float(M['region'][0]), float(M['press'][0])]

    def act(self, obs):
        x = self.features(obs)
        a = np.r_[0, 0, 0, 1.0] if self.readout is None else self.readout.predict(x[None, :8])[0]
        return a, x

    def fit(self, X, Y, g, rounds=None):
        w = share_weights(_flags_from(X, 8), self.recipe['shares'])
        self.readout = make_readout(self.recipe).fit(np.asarray(X[:, :8], float), Y, g, w)
        return self.readout


# ---------------------------------------------------------------- failure diagnostics
def hover_stats(outs):
    """Test episodes that ended still holding the cube: how long they held it near the target
    (within the release distance, 40 mm) and how low the cube got there (mm above resting)."""
    rows = []
    for o in outs:
        if o['outcome'] != 'still holding the cube' or o.get('obs') is None:
            continue
        ob = o['obs']
        near = (np.hypot(ob[:, 19], ob[:, 20]) < RELEASE_XY) & (ob[:, 10] > LIFT_Z)
        rows.append(dict(seed=o['seed'], steps_near_target=int(near.sum()),
                         lowest_mm_above_rest=round(1000 * float(ob[near, 10].min() - 0.025), 1) if near.any() else None,
                         final_dist_mm=round(float(o['place_error_mm']), 1)))
    return rows


def grip_agreement(outs):
    """Share of test steps where the executed grip (the decoded one) has the expert label's sign."""
    a = np.concatenate([o['A'][:, 3] > 0 for o in outs if o.get('A') is not None and len(o['A'])])
    y = np.concatenate([o['Y'][:, 3] > 0 for o in outs if o.get('Y') is not None and len(o['Y'])])
    return round(float(np.mean(a == y)), 3) if len(a) else None


# ---------------------------------------------------------------- week 9's run, for the warm start
def week9_warm(week9_dir, k=K_WIDE):
    """Week 9's connectome C4 run (its DAgger checkpoint, record and pool) in week9_dir, or None.
    With it, the connectome's C4 readout is refitted from week 9's data (exactly week 9's final
    readout) instead of being retrained: no brain time, and the test is of week 9's readout itself."""
    d = Path(week9_dir)
    if not d.is_dir():
        return None
    for p in sorted(d.glob('week9_*.pkl')):
        if p.stem.endswith(('_dagger', 'references')):
            continue
        try:
            r = pickle.load(open(p, 'rb'))
        except Exception:
            continue
        if not isinstance(r, dict) or 'random' in str(r.get('name', '')).lower():
            continue
        c = r.get('cand') or {}
        if not (c.get('labels') == 'B' and c.get('select') == 'all' and c.get('weight')):
            continue
        npz, log = p.with_name(p.stem + '_dagger.npz'), p.with_name(p.stem + '_dagger_log.json')
        if npz.exists() and log.exists():
            return dict(name=r['name'], pkl=p, npz=npz, log=log, record=np.asarray(r['record']),
                        pool_cols=np.asarray(r['pool_cols']), readout=r.get('readout'),
                        summary=r.get('summary'), k=k)
    return None


# ---------------------------------------------------------------- one network and recipe, end to end
def run_network10(name, sub, inputs, pool, env, obs_names, cfg, out, w_scale, recipe_key, resume=True,
                  extra_record=(), brain_ms=80, window_ms=80, build=_build_brain, warm=None):
    """recipe_key: 'C4', 'C6' or 'G'. warm: week9_warm(...) to reuse week 9's DAgger data (C4 only):
    then the policy is week 9's own (Week9BrainPolicy, 'all', weighted) on week 9's record."""
    out = Path(out); tag = slug(name)
    final = out / f'week10_{tag}.pkl'
    if resume and final.exists():
        done = pickle.load(open(final, 'rb'))
        if done.get('test_seeds') == list(cfg['test_seeds']):
            print(f"  loaded {final.name}")
            return done
    recipe = RECIPES[recipe_key]
    ex = EXPERTS[recipe['labels']]
    assert warm is None or recipe_key == 'C4', "the warm start is week 9's C4 readout"
    T0 = time.perf_counter()
    pool_env = EnvPool(env)
    ck_npz, ck_log = out / f'week10_{tag}_dagger.npz', out / f'week10_{tag}_dagger_log.json'
    if warm is not None:                  # read week 9's checkpoint in place (it is ~0.5 GB): no copy
        record, pool_cols = warm['record'], warm['pool_cols']
        ck_npz, ck_log = Path(warm['npz']), Path(warm['log'])
    else:
        record = np.unique(np.r_[np.asarray(pool), np.asarray(extra_record, dtype=np.int64)])
        pool_cols = np.searchsorted(record, np.asarray(pool))
    eng, brain = build(sub, inputs, record, cfg)
    max_batch = cfg.get('max_batch', 64) if getattr(eng, 'kind', 'torch') == 'torch' else cfg.get('cpu_batch', 1)
    k = cfg['k']
    if warm is not None and warm.get('readout') is not None:
        k = len(warm['readout']['cols'])  # week 9's own readout size, whatever cfg says
    common = dict(pool=pool_cols, k=k, brain_ms=brain_ms, window_ms=window_ms, name=name, max_batch=max_batch)
    if warm is not None:
        pol = Week9BrainPolicy(brain, w_scale, obs_names, select='all', weight=True, **common)
    else:
        pol = Week10BrainPolicy(brain, w_scale, obs_names, recipe=recipe, **common)
    res = dict(name=name, recipe=recipe_key, recipe_what=recipe['what'], w_scale=w_scale,
               timing=[brain_ms, window_ms], pool_cols=pool_cols, record=record,
               warm_from=str(warm['pkl']) if warm else None, test_seeds=list(cfg['test_seeds']))
    print(f"{name}: recipe {recipe_key} ({recipe['what']}); w_scale {w_scale:.3f}; "
          f"{'week 9 DAgger data (refit, no new rounds)' if warm else 'DAgger'}; {len(pool_cols):,} neurons in the pool")

    def checkpoint(r, data, log):
        if warm is not None:              # never write into week 9's (read-only) files
            return
        np.savez(ck_npz, **{kk: np.concatenate(data[kk]) for kk in ('X', 'Y', 'g', 'round', 'obs')})
        ck_log.write_text(json.dumps(log, default=float))

    resume_from = None
    if (resume or warm is not None) and ck_npz.exists() and ck_log.exists():
        d0 = np.load(ck_npz); resume_from = ({kk: d0[kk] for kk in d0.files}, json.loads(ck_log.read_text()))
    t0 = time.perf_counter()
    data, log = dagger_pp(pool_env, pol, ex, cfg['train_seeds'], cfg['betas'], on_round=checkpoint,
                          resume=resume_from)
    res['dagger'] = dict(log=log, rows=int(len(data['Y'])), wall_min=(time.perf_counter() - t0) / 60,
                         alpha=pol.readout.alpha)
    res['readout'] = dict(cols=pol.cols, model=pol.readout, alpha=pol.readout.alpha)
    if warm is not None and warm.get('readout') is not None:
        w9 = warm['readout']
        same_cols = bool(np.array_equal(np.asarray(w9['cols']), np.asarray(pol.cols)))
        dW = float(np.abs(np.asarray(w9['W']) - pol.readout.W).max()) if same_cols else None
        res['warm_check'] = dict(same_neurons=same_cols, max_weight_diff=dW)
        print(f"  week 9's readout rebuilt: same 3,000 neurons {same_cols}, largest weight difference {dW}")
    del data

    t0 = time.perf_counter()
    outs, s = evaluate_pp(pool_env, pol, ex, cfg['test_seeds'], label=name, record=True)
    res.update(outs=light(outs), summary=s, test_wall_min=(time.perf_counter() - t0) / 60,
               trap=trap_stats(outs), hover=hover_stats(outs), grip_agreement=grip_agreement(outs))
    if cfg.get('cut_seeds'):
        pol.feedback = False
        co, cs = evaluate_pp(pool_env, pol, ex, cfg['cut_seeds'], label=f'{name}, feedback cut')
        pol.feedback = True
        res['cut'] = dict(outs=light(co), summary=cs)
    order = sorted(outs, key=lambda o: (-o['success'], -o['zone'], -o['lifted'], o['seed']))
    vid = order[:cfg.get('n_video', 1)]
    fails = [o for o in outs if not o['success']]
    if fails and fails[0]['seed'] not in {o['seed'] for o in vid}:     # no success: don't show it twice
        vid.append(fails[0])
    res['video_eps'] = [dict(rec=o['rec'], seed=o['seed'], success=o['success'], outcome=o['outcome'], net=name)
                        for o in vid]
    del outs
    res['wall_min'] = (time.perf_counter() - T0) / 60
    for o in res['outs']:
        o.pop('obs', None); o.pop('Y', None); o.pop('A', None); o.pop('phase', None)
    pickle.dump(res, open(final, 'wb'))
    print(f"  done in {res['wall_min']:.0f} min")
    del pol, brain, eng
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return res


# ---------------------------------------------------------------- no-brain controls
REFERENCES = ('expert, fix B', 'MLP on 8 signals', 'linear on 8 signals', 'linear on 8 signals, C4 weights',
              'linear on 8 signals, C6 weights', 'gated linear on 8 signals (G)')
CONTROL_OF = {'C4': 'linear on 8 signals, C4 weights', 'C6': 'linear on 8 signals, C6 weights',
              'G': 'gated linear on 8 signals (G)'}


def run_references10(env, obs_names, cfg, out, resume=True, which=REFERENCES):
    """Every no-brain policy trained exactly as the brain (DAgger, same seeds, betas, noise),
    tested on the same fresh seeds. Saved after each policy, so a restart skips finished ones."""
    out = Path(out); path = out / f"week10_references_r{len(cfg['train_seeds'])}x{len(cfg['train_seeds'][0])}.pkl"
    res = pickle.load(open(path, 'rb')) if (resume and path.exists()) else {}
    ex = EXPERTS['B']
    for nm in which:
        if nm in res and res[nm].get('test_seeds') == list(cfg['test_seeds']):
            continue
        t0 = time.perf_counter()
        if nm.startswith('expert'):
            p, log = ExpertPolicy(ex, obs_names), None
        else:
            if nm == 'MLP on 8 signals':
                p = SignalPolicy(obs_names, 'mlp', name=nm)
            elif nm == 'linear on 8 signals':
                p = SignalPolicy(obs_names, 'ridge', name=nm)
            else:
                key = next(k for k, v in CONTROL_OF.items() if v == nm)
                p = FlagSignalPolicy(obs_names, RECIPES[key], nm)
            _, log = dagger_pp(env, p, ex, cfg['train_seeds'], cfg['betas'], verbose=False)
        outs, s = evaluate_pp(env, p, ex, cfg['test_seeds'], label=nm)
        res[nm] = dict(summary=s, outs=light(outs), log=log, test_seeds=list(cfg['test_seeds']),
                       wall_min=(time.perf_counter() - t0) / 60)
        for o in res[nm]['outs']:
            o.pop('obs', None); o.pop('Y', None); o.pop('A', None); o.pop('phase', None)
        pickle.dump(res, open(path, 'wb'))
    return res


# ---------------------------------------------------------------- the comparisons (rules fixed before the run)
def _mannwhitney(a, b):
    """Two-sided Mann-Whitney U p-value (normal approximation with tie correction)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    allv = np.r_[a, b]
    order = allv.argsort(kind='stable')
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    _, cnt = np.unique(allv, return_counts=True)
    n = n1 + n2
    var = n1 * n2 / 12 * ((n + 1) - (cnt ** 3 - cnt).sum() / (n * (n - 1)))
    if var <= 0:
        return 1.0
    z = (abs(u - n1 * n2 / 2) - 0.5) / np.sqrt(var)
    from math import erfc, sqrt
    return float(min(1.0, erfc(max(z, 0) / sqrt(2))))


def secondary(outs_a, outs_b, min_success=10):
    """Week 7's rule: placement error and time to complete are compared only if both policies
    succeed on >= min_success seeds. Medians and a two-sided Mann-Whitney p on the successes."""
    sa = [o for o in outs_a if o['success']]; sb = [o for o in outs_b if o['success']]
    if len(sa) < min_success or len(sb) < min_success:
        return dict(compared=False, why=f"needs >= {min_success} successes each ({len(sa)} vs {len(sb)})")
    pa, pb = [o['place_error_mm'] for o in sa], [o['place_error_mm'] for o in sb]
    ta, tb = [0.04 * o['steps'] for o in sa], [0.04 * o['steps'] for o in sb]
    return dict(compared=True,
                placement_mm=dict(a=round(float(np.median(pa)), 1), b=round(float(np.median(pb)), 1),
                                  p=round(_mannwhitney(pa, pb), 4)),
                time_s=dict(a=round(float(np.median(ta)), 2), b=round(float(np.median(tb)), 2),
                            p=round(_mannwhitney(ta, tb), 4)))


def compare10(nets, refs, conn='connectome, C4', rand='random network, C4', conn_g='connectome, G',
              rand_g='random network, G'):
    """nets: {name: run_network10 result}. refs: run_references10 result.
    1. Headline: the connectome with C4 on fresh seeds; pass >= 40/50.
    2. Wiring (pre-registered, week 7's rule): connectome C4 vs random network C4, paired by seed,
       milestone by milestone from the hardest (success); the first with exact McNemar p < 0.05
       decides. Placement and time compared only if both have >= 10 successes. Repeated for G
       (secondary) if the random network was also run with G.
    3. Toward 50/50: each connectome readout's successes, paired against C4 (McNemar). The best is
       the one with the most successes (ties: the simpler, C4 < C6 < G). A readout's result is the
       brain's only if its own no-brain control (the same readout on the 8 noise-free signals,
       trained the same way) stays below the pass mark (< 40/50)."""
    out = dict()
    s = nets[conn]['summary'] if conn in nets else None
    if s:
        out['headline'] = dict(success=s['success'], n=s['n'], ci=s['ci'], passed=s['success'] >= PASS * s['n'] / 50)
    for key, (a, b) in dict(wiring=(conn, rand), wiring_G=(conn_g, rand_g)).items():
        if a in nets and b in nets:
            v, pm = verdict(nets[a]['outs'], nets[b]['outs'])
            out[key] = dict(verdict=v, milestones=pm, secondary=secondary(nets[a]['outs'], nets[b]['outs']),
                            success=(nets[a]['summary']['success'], nets[b]['summary']['success']))
    rows = {}
    for rk in SIMPLER:
        nm = f'connectome, {rk}'
        if nm not in nets:
            continue
        r = nets[nm]
        ctl = refs.get(CONTROL_OF[rk], {}).get('summary') if refs else None
        row = dict(success=r['summary']['success'], n=r['summary']['n'],
                   control=CONTROL_OF[rk], control_success=None if ctl is None else ctl['success'],
                   control_n=None if ctl is None else ctl['n'],
                   brain_needed=None if ctl is None else bool(ctl['success'] < PASS * ctl['n'] / 50))
        if rk != 'C4' and conn in nets:
            pm = paired_milestones(r['outs'], nets[conn]['outs'])['success']
            row.update(only_this=pm['only_a'], only_C4=pm['only_b'], p_vs_C4=pm['p'])
        rows[rk] = row
    out['readouts'] = rows
    if rows:
        ok = [k for k in SIMPLER if k in rows and rows[k]['brain_needed'] is not False]
        cand = ok or [k for k in SIMPLER if k in rows]
        top = max(rows[k]['success'] for k in cand)
        best = next(k for k in SIMPLER if k in cand and rows[k]['success'] == top)
        out['best'] = dict(readout=best, success=top, n=rows[best]['n'], target_50=bool(top == rows[best]['n']),
                           counted_only_brain_needed=bool(ok))
    return out


def describe10(cmp):
    """Plain-language lines for the notebook's output."""
    L = []
    h = cmp.get('headline')
    if h:
        L.append(f"HEADLINE: connectome, C4 on fresh seeds: {h['success']}/{h['n']} (95% CI {h['ci'][0]:.0%}-"
                 f"{h['ci'][1]:.0%}); pass (>= {PASS}/50): {'YES' if h['passed'] else 'NO'}")
    for key, lab in (('wiring', 'WIRING (pre-registered, C4)'), ('wiring_G', 'WIRING, gated readout (secondary)')):
        if key in cmp:
            w = cmp[key]
            L.append(f"{lab}: {w['verdict']}. Success {w['success'][0]} vs {w['success'][1]}.")
            sec = w['secondary']
            if sec['compared']:
                L.append(f"   placement {sec['placement_mm']['a']} vs {sec['placement_mm']['b']} mm "
                         f"(p = {sec['placement_mm']['p']}), time {sec['time_s']['a']} vs {sec['time_s']['b']} s "
                         f"(p = {sec['time_s']['p']})")
            else:
                L.append(f"   placement and time not compared: {sec['why']}")
    for k, r in cmp.get('readouts', {}).items():
        extra = f"; vs C4: {r['only_this']} seeds only {k}, {r['only_C4']} only C4 (p = {r['p_vs_C4']:.3g})" \
            if 'p_vs_C4' in r else ''
        ctl = '' if r['control_success'] is None else \
            f"; no-brain control ({r['control']}) {r['control_success']}/{r['control_n']} -> brain needed: {r['brain_needed']}"
        L.append(f"READOUT {k}: {r['success']}/{r['n']}{extra}{ctl}")
    if 'best' in cmp:
        b = cmp['best']
        L.append(f"BEST CONNECTOME READOUT: {b['readout']} at {b['success']}/{b['n']}; 50/50 target reached: "
                 f"{'YES' if b['target_50'] else 'NO'}")
    return L
