"""Week 9: finish the retreat.

Week 8b: the 3,000-neuron readout completes 25/50, and all 25 failures are one thing: the cube is
put down and released, then the decoded dz sits at about 0 +- 10 mm/step and the open gripper
hovers just above the cube until time runs out (success needs it 6 cm clear). The rest of the task
works. Diagnosis: a cliff in the label at the release (dz 0 while the fingers open, +20 mm/step
once they pass 70 mm), between states a linear readout of noisy rates can't tell apart well.

Candidate fixes. Every one keeps the frozen connectome and a LINEAR readout; they differ only in
the teacher's labels, which neurons the readout uses, and how rows are weighted in the fit:

  B2         graded retreat labels (sim/expert9.py): dz ramps 0 -> +20 as the fingers open.
  reselect   the 3,000 neurons are re-chosen at every DAgger refit on ALL data collected so far,
             not once on round 0 (the expert-driven round has ~4 retreat steps per episode; the
             later rounds hold the states the brain actually gets stuck in).
  weight     rows in the release region (cube on the table at the target) carry END_SHARE of the
             total weight in the ridge fit and in the neuron selection (if they carry less).

  Step 1 (offline, no brain time): week 8b's saved DAgger data (rates of all 9,500 recorded neurons
  at every visited state, rounds 0-3) refit with each candidate, out of episode, and scored on the
  states that matter: the TRAP (released, gripper open and still low: must go up), the PRESS
  (holding the cube in the last 4 cm above the table: must go down), and everything else.
  Step 2: the closed loop with the candidate the rule picks. Week 8b's pipeline otherwise unchanged.

  WRidge          weighted ridge, one penalty per output group (as RidgeMulti), solved by eigh of
                  the Gram matrix (same solution as the SVD, several times faster at 14k x 3k)
  region_masks    trap / press / release / release-region rows, from the observations only
  offline9        the candidate table;  choose9  the rule (fixed before the run)
  Week9BrainPolicy, run_network9   the closed loop with a candidate's settings
"""
import gc, json, pickle, time
from pathlib import Path

import numpy as np

from expert import CUBE_HALF, HELD_W, STRADDLE_XY
from .expert8 import RELEASE_DZ, RELEASE_XY, expert8
from .expert9 import expert9, W_FULL
from .pickplace import SIGNALS8, ExpertPolicy, dagger_pp, evaluate_pp, EnvPool
from .readout import ALPHAS, _folds, r2
from .week7 import light, slug
from .week8 import Week8BrainPolicy

K_WIDE = 3000
END_SHARE = 1 / 3          # release-region rows carry at least this share of the fit's weight
CLEAR_Z = 0.06             # panda_env.info(): gripper this far above the cube counts as clear
PRESS_ABOVE = 0.040        # press rows: cube centre within this of the release region's top
DEAD = 0.003               # m/step: decoded dz beyond +-3 mm/step counts as a clear up / down

EXPERTS = {'B': expert8, 'B2': expert9}

# Candidates, simplest first (one change each, then combinations). The rule prefers the simpler
# of two that score within TIE of each other.
CANDIDATES = {
    'C0 week 8b (round-0 selection)': dict(labels='B', select='round0', weight=False),
    'C1 reselect on all data': dict(labels='B', select='all', weight=False),
    'C2 weight the release region': dict(labels='B', select='round0', weight=True),
    'C3 B2 graded retreat': dict(labels='B2', select='round0', weight=False),
    'C4 reselect + weight': dict(labels='B', select='all', weight=True),
    'C5 B2 + reselect + weight': dict(labels='B2', select='all', weight=True),
}
# Reported, never chosen: two linear motion readouts switched by the decoded grip (one multiplicative
# interaction in the readout). It tells week 10 how much a slightly nonlinear readout would add.
REPORT_ONLY = {'G gated by decoded grip (not linear)': dict(labels='B', select='all', weight=False, gated=True)}

RULE = dict(end_min=0.85, r2_drop=0.05, grip_drop=0.02, tie=0.03, partial_gain=0.10)


# ---------------------------------------------------------------- the readout
class WRidge:
    """Weighted ridge, standardised features, one penalty per group of output columns (motion and
    grip, as RidgeMulti), chosen by grouped CV with weighted squared error. w=None is plain ridge.
    Solved through the eigendecomposition of the (weighted) Gram matrix."""

    def __init__(self, col_groups=((0, 1, 2), (3,)), alphas=ALPHAS, n_folds=3, seed=0):
        self.col_groups = [list(c) for c in col_groups]
        self.alphas, self.n_folds, self.seed = np.asarray(alphas), n_folds, seed

    @staticmethod
    def _prep(X, Y, w):
        sw = w.sum()
        mu = (w @ X) / sw
        sd = np.sqrt(np.maximum((w @ (X - mu) ** 2) / sw, 0))
        ok = sd > 1e-9
        ym = (w @ Y) / sw
        return mu, sd, ok, ym

    def _path(self, Z, Yc, w):
        r = np.sqrt(w)[:, None]
        Zw, Yw = Z * r, Yc * r
        lam, V = np.linalg.eigh(Zw.T @ Zw)
        lam = np.clip(lam, 0, None)
        B = V.T @ (Zw.T @ Yw)
        return V, lam, B

    @staticmethod
    def _coef(V, lam, B, a):
        return V @ (B / (lam + a)[:, None])

    def fit(self, X, Y, groups, w=None):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        w = np.ones(len(X)) if w is None else np.asarray(w, float)
        w = w * (len(w) / w.sum())
        err = np.zeros((len(self.col_groups), len(self.alphas)))
        if len(np.unique(groups)) >= self.n_folds and len(self.alphas) > 1:
            for f in _folds(groups, self.n_folds, self.seed + 1):
                mu, sd, ok, ym = self._prep(X[~f], Y[~f], w[~f])
                Za, Zb = (X[~f][:, ok] - mu[ok]) / sd[ok], (X[f][:, ok] - mu[ok]) / sd[ok]
                V, lam, B = self._path(Za, Y[~f] - ym, w[~f])
                for i, a in enumerate(self.alphas):
                    e = w[f][:, None] * (Zb @ self._coef(V, lam, B, a) + ym - Y[f]) ** 2
                    for gi, cols in enumerate(self.col_groups):
                        err[gi, i] += e[:, cols].sum()
            self.alpha = [float(self.alphas[int(np.argmin(e))]) for e in err]
        else:
            self.alpha = [float(self.alphas[len(self.alphas) // 2])] * len(self.col_groups)
        mu, sd, self.ok, self.ym = self._prep(X, Y, w)
        self.mu, self.sd = mu[self.ok], sd[self.ok]
        Z = (X[:, self.ok] - self.mu) / self.sd
        V, lam, B = self._path(Z, Y - self.ym, w)
        self.W = np.zeros((Z.shape[1], Y.shape[1]))
        for a, cols in zip(self.alpha, self.col_groups):
            self.W[:, cols] = self._coef(V, lam, B[:, cols], a)
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        return ((X[:, self.ok] - self.mu) / self.sd) @ self.W + self.ym


class GatedRidge:
    """REPORT ONLY. Grip from one ridge on all rows; motion from one of two ridges (fitted on rows
    whose grip label is open / closed), switched by the decoded grip."""

    def fit(self, X, Y, groups, w=None):
        self.all = WRidge().fit(X, Y, groups, w)
        op = Y[:, 3] > 0
        self.open = WRidge().fit(X[op], Y[op], groups[op], None if w is None else w[op])
        self.closed = WRidge().fit(X[~op], Y[~op], groups[~op], None if w is None else w[~op])
        self.alpha = self.all.alpha
        return self

    def predict(self, X):
        P = self.all.predict(X)
        op = P[:, 3] > 0
        if op.any():
            P[op, :3] = self.open.predict(X[op])[:, :3]
        if (~op).any():
            P[~op, :3] = self.closed.predict(X[~op])[:, :3]
        return P


def informative_w(X, Y, k, w=None):
    """`pickplace.informative` (max |Pearson r| with any output), with row weights."""
    w = np.ones(len(X)) if w is None else np.asarray(w, float)
    sw = w.sum()
    Xc = X - (w @ X) / sw
    Yc = Y - (w @ Y) / sw
    sx = np.sqrt(w @ Xc ** 2); sy = np.sqrt(w @ Yc ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = ((Xc * w[:, None]).T @ Yc) / np.outer(sx, sy)
    score = np.nan_to_num(np.abs(r)).max(1)
    order = np.argsort(-score, kind='stable')
    return np.sort(order[:k]), score


# ---------------------------------------------------------------- which states are which
def region_masks(obs):
    """From the observations only (the same for both label sets):
      region   cube on the table at the target (expert8's release branch: within 40 mm, cube centre
               < 10 mm above resting height): the rows `weight` up-weights
      trap     region, fingers open past 70 mm (released), gripper not yet 6 cm clear: must go up
      release  region, fingers still around the cube (<= 70 mm, straddled)
      press    holding the cube near the target, cube 10-50 mm above resting height: must go down"""
    obs = np.atleast_2d(obs)
    tcp_z, width, cube_z = obs[:, 2], obs[:, 6], obs[:, 10]
    rel = obs[:, 16:19]
    d_goal = np.hypot(obs[:, 19], obs[:, 20])
    straddled = (np.hypot(rel[:, 0], rel[:, 1]) < STRADDLE_XY) & (np.abs(rel[:, 2]) < 0.025)
    near = d_goal < RELEASE_XY
    region = near & (cube_z < CUBE_HALF + RELEASE_DZ)
    trap = region & (width > W_FULL) & (tcp_z - cube_z < CLEAR_Z)
    release = region & (width <= W_FULL) & straddled
    press = near & ~region & (cube_z < CUBE_HALF + RELEASE_DZ + PRESS_ABOVE) & (width < HELD_W) & straddled
    return dict(region=region, trap=trap, release=release, press=press)


def end_weights(flag, share=END_SHARE):
    """Weights giving the flagged rows `share` of the total, if they have less; mean 1."""
    flag = np.asarray(flag, bool)
    n, ne = len(flag), int(flag.sum())
    if ne == 0 or ne >= share * n:
        return np.ones(n)
    w = np.where(flag, share / (1 - share) * (n - ne) / ne, 1.0)
    return w * (n / w.sum())


def relabel(obs, which):
    return np.array([EXPERTS[which](o)[0] for o in obs])


# ---------------------------------------------------------------- step 1: offline
def _fit_candidate(spec, Xp, Y, g, rnd, region, k):
    """Choose columns and fit on training rows. Returns (cols, model)."""
    w = end_weights(region) if spec['weight'] else None
    if spec['select'] == 'round0':
        m0 = rnd == 0
        cols = informative_w(Xp[m0], Y[m0], k)[0] if k < Xp.shape[1] else np.arange(Xp.shape[1])
    else:
        cols = informative_w(Xp, Y, k, w)[0] if k < Xp.shape[1] else np.arange(Xp.shape[1])
    model = (GatedRidge() if spec.get('gated') else WRidge()).fit(Xp[:, cols], Y, g, w)
    return cols, model


def score_rows(P, Y, masks):
    """The candidate's metrics from out-of-episode predictions P against its own labels Y."""
    rr = r2(Y[:, :3], P[:, :3])
    lab = Y[:, 3] > 0
    bal = 0.5 * (np.mean((P[:, 3] > 0)[lab]) + np.mean((P[:, 3] <= 0)[~lab]))
    trap = masks['trap'] & (Y[:, 2] > DEAD)
    press = masks['press'] & (Y[:, 2] < -DEAD)
    rel = masks['release']
    nan = float('nan')
    return {'dx R2': round(float(rr[0]), 3), 'dy R2': round(float(rr[1]), 3), 'dz R2': round(float(rr[2]), 3),
            'grip bal. acc.': round(float(bal), 3),
            'trap: dz up': round(float(np.mean(P[trap, 2] > DEAD)), 3) if trap.any() else nan,
            'trap: mean dz (mm/step)': round(1000 * float(np.mean(P[trap, 2])), 1) if trap.any() else nan,
            'trap: grip open': round(float(np.mean(P[trap, 3] > 0)), 3) if trap.any() else nan,
            'press: dz down': round(float(np.mean(P[press, 2] < -DEAD)), 3) if press.any() else nan,
            'release: grip open': round(float(np.mean(P[rel, 3] > 0)), 3) if rel.any() else nan}


def offline9(d, pool_cols, candidates=None, k=K_WIDE, n_folds=5, seed=0, verbose=True):
    """d: week 8b's DAgger checkpoint (X, Y, g, round, obs). pool_cols: the readout pool's columns in
    d['X'] (week 8b: the active neurons + all DNs). Out of episode (n_folds folds by whole episode):
    for each candidate, columns and readout fitted on the training episodes only, predictions for
    the held-out ones. Returns (rows, counts)."""
    candidates = dict(CANDIDATES, **REPORT_ONLY) if candidates is None else candidates
    obs, g, rnd = d['obs'], d['g'], d['round']
    Xp = d['X'][:, np.asarray(pool_cols)].astype(np.float64)
    M = region_masks(obs)
    labels = {'B': np.asarray(d['Y'], float)}
    if any(c['labels'] == 'B2' for c in candidates.values()):
        labels['B2'] = relabel(obs, 'B2')
    folds = _folds(g, n_folds, seed)
    rows = []
    for name, spec in candidates.items():
        t0 = time.perf_counter()
        Y = labels[spec['labels']]
        P = np.zeros_like(Y)
        for f in folds:
            cols, model = _fit_candidate(spec, Xp[~f], Y[~f], g[~f], rnd[~f], M['region'][~f], k)
            P[f] = model.predict(Xp[f][:, cols])
        row = dict(candidate=name, **{kk: spec.get(kk) for kk in ('labels', 'select', 'weight')},
                   **score_rows(P, Y, M), s=round(time.perf_counter() - t0, 1))
        rows.append(row)
        if verbose:
            print(f"  {name}: trap up {row['trap: dz up']:.0%} (mean {row['trap: mean dz (mm/step)']:+.1f} mm/step), "
                  f"press down {row['press: dz down']:.0%}, dx/dy/dz R2 {row['dx R2']:.2f}/{row['dy R2']:.2f}/"
                  f"{row['dz R2']:.2f}, grip {row['grip bal. acc.']:.2f}  [{row['s']:.0f} s]")
    Yb = labels['B']
    trap = M['trap']
    counts = dict(rows=len(g), episodes=int(len(np.unique(g))), rounds=int(rnd.max()) + 1,
                  region=int(M['region'].sum()), trap=int(trap.sum()), press=int(M['press'].sum()),
                  release=int(M['release'].sum()),
                  trap_by_round=[int((trap & (rnd == r)).sum()) for r in range(int(rnd.max()) + 1)],
                  trap_label_up=round(float(np.mean(Yb[trap, 2] > DEAD)), 3) if trap.any() else None,
                  region_share=round(float(M['region'].mean()), 3))
    return rows, counts


def end_score(r):
    """The weaker of the two end phases: going up in the trap, going down in the press. A fix that
    lifts the gripper out of the trap by making it rise everywhere near the target scores low."""
    return float(min(r['trap: dz up'], r['press: dz down']))


def choose9(rows, rule=RULE):
    """The rule, fixed before the run. Baseline = C0 (week 8b's readout recipe).
    Guards: dx and dy R2 >= C0's - r2_drop (0.05), grip balanced accuracy >= C0's - grip_drop (0.02).
    ELIGIBLE: passes the guards and end_score = min(trap: dz up, press: dz down) >= end_min (0.85).
    Among eligible ones the highest end_score wins; one within `tie` (0.03) of it that is simpler
    (earlier in CANDIDATES) is taken instead. If none is eligible: the guard-passing candidate with
    the best end_score is run if it beats C0's by >= partial_gain (0.10) ('partial'); otherwise no
    closed loop ('stop'). Report-only rows are never chosen.
    Returns (decision, candidate name or None, why)."""
    order = list(CANDIDATES)
    R = {r['candidate']: r for r in rows if r['candidate'] in CANDIDATES}
    base = R[order[0]]
    sb = end_score(base)

    def guards(r):
        return (r['dx R2'] >= base['dx R2'] - rule['r2_drop'] and r['dy R2'] >= base['dy R2'] - rule['r2_drop']
                and r['grip bal. acc.'] >= base['grip bal. acc.'] - rule['grip_drop'])

    def desc(n):
        r = R[n]
        return f"trap up {r['trap: dz up']:.0%}, press down {r['press: dz down']:.0%}"
    ok = [n for n in order[1:] if n in R and guards(R[n])]
    elig = [n for n in ok if end_score(R[n]) >= rule['end_min']]
    if elig:
        best = max(end_score(R[n]) for n in elig)
        pick = next(n for n in elig if end_score(R[n]) >= best - rule['tie'])
        return 'closed loop', pick, f"{pick}: {desc(pick)} (C0: {desc(order[0])})"
    if ok:
        pick = max(ok, key=lambda n: end_score(R[n]))
        if end_score(R[pick]) >= sb + rule['partial_gain']:
            return 'partial', pick, (f"no candidate reaches {rule['end_min']:.0%} on both end phases; best is {pick}: "
                                     f"{desc(pick)} (C0: {desc(order[0])}): run it")
    return 'stop', None, ("no linear candidate clearly improves the end phases: no closed loop. Week 10: the "
                          "readout itself (see the report-only gated row)")


# ---------------------------------------------------------------- step 2: the closed loop
class Week9BrainPolicy(Week8BrainPolicy):
    """Week 8b's policy (80/80 timing, 3,000 neurons, ridge) with a candidate's settings:
    select 'round0' (chosen once on round 0, as week 8b) or 'all' (re-chosen at every refit on all
    data so far), weight (release-region rows up-weighted). act() appends one column to the logged
    features, the release-region flag of the state, so that fit() can weight rows; the readout
    never sees it (it is outside the pool)."""

    def __init__(self, *a, select='round0', weight=False, **kw):
        super().__init__(*a, **kw)
        self.select, self.weight = select, bool(weight)

    def act(self, obs_list):
        A, X = super().act(obs_list)
        flag = region_masks(np.array(obs_list))['region'].astype(float)
        return A, np.c_[X, flag]

    def fit(self, X, Y, g, rounds=None):
        flag = X[:, -1] > 0.5
        w = end_weights(flag) if self.weight else None
        Xp = X[:, self.pool].astype(float)
        if self.k is None or self.k >= len(self.pool):
            sel = np.arange(len(self.pool))
        elif self.select == 'all' or self.cols is None:
            sel = informative_w(Xp, Y, self.k, w if self.select == 'all' else None)[0]
        else:
            sel = None
        if sel is not None:
            self.cols = self.pool[sel]
        self.readout = WRidge().fit(X[:, self.cols], Y, g, w)
        Wm = np.zeros((len(self.cols), 4))
        Wm[self.readout.ok] = self.readout.W / self.readout.W.std(0).clip(1e-12)
        top = np.argsort(-np.abs(Wm).max(1))[:100]
        top = top[np.lexsort((-np.abs(Wm[top]).max(1), np.abs(Wm[top]).argmax(1)))]
        self.raster_cols = self.cols[top]
        return self.readout


def _build_brain(sub, inputs, record, cfg):
    """The week 7/8 engine and encoder for this network (replaced by a stand-in in sandbox tests)."""
    from .interface import Encoder
    from .batch7 import make_engine, BatchBrain
    from .network7 import wiring_labels
    labels = wiring_labels(sub, inputs, 2 * len(SIGNALS8))
    enc = Encoder(inputs, labels, SIGNALS8, kind='wiring')
    eng = make_engine(sub, inputs, cfg.get('use_gpu', False), cfg.get('dtype', 'float32'),
                      torch_device=cfg.get('torch_device'))
    return eng, BatchBrain(eng, enc, record)


def trap_stats(outs):
    """How the test episodes that put the cube down but didn't finish ended: steps spent in the trap
    and the gripper's highest point above the cube after releasing (mm)."""
    rows = []
    for o in outs:
        if o['outcome'] != 'put down in zone, not finished' or o.get('obs') is None:
            continue
        ob = o['obs']; m = region_masks(ob)
        rel = m['region'] & (ob[:, 6] > W_FULL)
        rows.append(dict(seed=o['seed'], trap_steps=int(m['trap'].sum()),
                         max_clear_mm=round(1000 * float((ob[rel, 2] - ob[rel, 10]).max()), 1) if rel.any() else None))
    return rows


def run_network9(name, sub, inputs, pool, env, obs_names, cfg, out, w_scale, cand, resume=True,
                 extra_record=(), brain_ms=80, window_ms=80, build=_build_brain):
    """week8.run_network8 with a week 9 candidate (dict: labels, select, weight) in place of a fix."""
    out = Path(out); tag = slug(name)
    final = out / f'week9_{tag}.pkl'
    if resume and final.exists():
        done = pickle.load(open(final, 'rb'))
        if len(done['dagger']['log']) == len(cfg['train_seeds']):
            print(f"  loaded {final.name}")
            return done
    ex = EXPERTS[cand['labels']]
    T0 = time.perf_counter()
    pool_env = EnvPool(env)
    record = np.unique(np.r_[np.asarray(pool), np.asarray(extra_record, dtype=np.int64)])
    pool_cols = np.searchsorted(record, np.asarray(pool))
    eng, brain = build(sub, inputs, record, cfg)
    max_batch = cfg.get('max_batch', 64) if getattr(eng, 'kind', 'torch') == 'torch' else cfg.get('cpu_batch', 1)
    pol = Week9BrainPolicy(brain, w_scale, obs_names, pool=pool_cols, k=cfg['k'], brain_ms=brain_ms,
                           window_ms=window_ms, name=name, max_batch=max_batch,
                           select=cand['select'], weight=cand['weight'])
    res = dict(name=name, cand=dict(cand), w_scale=w_scale, timing=[brain_ms, window_ms], pool_cols=pool_cols,
               record=record)
    print(f"{name}: labels {cand['labels']}, neurons chosen on {cand['select']}, release region weighted "
          f"{cand['weight']}; w_scale {w_scale:.3f}, {brain_ms} ms brain per step, {window_ms} ms window")

    ck_npz, ck_log = out / f'week9_{tag}_dagger.npz', out / f'week9_{tag}_dagger_log.json'

    def checkpoint(r, data, log):
        np.savez(ck_npz, **{kk: np.concatenate(data[kk]) for kk in ('X', 'Y', 'g', 'round', 'obs')})
        ck_log.write_text(json.dumps(log, default=float))

    resume_from = None
    if resume and ck_npz.exists() and ck_log.exists():
        d0 = np.load(ck_npz); resume_from = ({kk: d0[kk] for kk in d0.files}, json.loads(ck_log.read_text()))
    t0 = time.perf_counter()
    data, log = dagger_pp(pool_env, pol, ex, cfg['train_seeds'], cfg['betas'], on_round=checkpoint,
                          resume=resume_from)
    res['dagger'] = dict(log=log, rows=int(len(data['Y'])), wall_min=(time.perf_counter() - t0) / 60,
                         alpha=pol.readout.alpha)
    res['readout'] = dict(cols=pol.cols, W=pol.readout.W, alpha=pol.readout.alpha)
    del data

    t0 = time.perf_counter()
    outs, s = evaluate_pp(pool_env, pol, ex, cfg['test_seeds'], label=name, record=True)
    res.update(outs=light(outs), summary=s, test_wall_min=(time.perf_counter() - t0) / 60,
               trap=trap_stats(outs))
    if cfg.get('cut_seeds'):
        pol.feedback = False
        co, cs = evaluate_pp(pool_env, pol, ex, cfg['cut_seeds'], label=f'{name}, feedback cut')
        pol.feedback = True
        res['cut'] = dict(outs=light(co), summary=cs)
    order = sorted(outs, key=lambda o: (-o['success'], -o['zone'], -o['lifted'], o['seed']))
    vid = order[:cfg.get('n_video', 2)]
    fails = [o for o in outs if not o['success']]
    if fails:
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


# ---------------------------------------------------------------- no-brain references for the chosen labels
def run_references9(env, obs_names, cfg, labels):
    """Expert, linear and MLP on the 8 noise-free signals, trained exactly as the brain (DAgger)."""
    from .pickplace import SignalPolicy
    ex = EXPERTS[labels]
    res = {}
    outs, s = evaluate_pp(env, ExpertPolicy(ex, obs_names), ex, cfg['test_seeds'], label=f'expert {labels}')
    res[f'expert, {labels}'] = dict(summary=s)
    for kind, nm in (('ridge', 'linear on 8 signals'), ('mlp', 'MLP on 8 signals')):
        p = SignalPolicy(obs_names, kind, name=nm)
        dagger_pp(env, p, ex, cfg['train_seeds'], cfg['betas'], verbose=False)
        outs, s = evaluate_pp(env, p, ex, cfg['test_seeds'], label=f'{nm}, {labels}')
        res[f'{nm}, {labels}'] = dict(summary=s)
    return res
