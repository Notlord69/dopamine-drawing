"""Week 10b: is week 10's random network typical? Three more shuffles, then the wiring verdict.

Week 10 (fresh seeds 93000-93049, readout C4): connectome 46/50, random network (week 7's shuffle,
seed 0) 50/50; no significant difference on any milestone (McNemar), and the random network placed
more precisely (5.4 vs 14.3 mm, p < 0.0001). That is one random network. Here the same comparison
is repeated with three more shuffles (seeds 1, 2, 3 of network7.random_network: every connection's
target permuted; each neuron's in- and out-degree, every connection's synapse count and sign kept).

Each shuffle gets exactly the rules week 7 gave seed 0, applied to its own wiring and activity:
  * w_scale: week 5/7's activity match (share of non-input neurons firing >= 1 Hz on week 7's 17
    reference states equals the connectome's), the batched grid search from the connectome's w_scale;
  * encoder channels from its own wiring; readout pool: its own active neurons (week 8b's rule) + all DNs;
  * readout C4 (sim/week10.py), DAgger 4 x 30 on seeds 64000+, test on 93000-93049, feedback cut.
A check first: the same search on seed 0 should land near week 7's 1.13.

  activity_target     the connectome's share of non-input neurons firing (the matching target)
  match_shuffle       week 7's search for one network; saved, so a restart skips it
  load_week10         week 10's connectome C4 and random network (seed 0) results
  compare10b          per-network verdicts (week 7's rule) and the final rule across all shuffles
"""
import json, pickle
from pathlib import Path

import numpy as np

from .week7 import frac_firing_batch, match_w_scale_batch, verdict
from .week10 import secondary, PASS

SHUFFLES = (1, 2, 3)


def activity_target(brain, w_scale, states, others_local):
    """Share of non-input neurons firing >= 1 Hz at w_scale (week 7's frac_firing_batch, one trial)."""
    f, _ = frac_firing_batch(brain, [w_scale], states, others_local)
    return float(f[0])


def match_shuffle(brain, w_ref, target, states, others_local, path, resume=True):
    """Week 7's match_w_scale_batch for one network; returns dict(w_scale, frac_firing, tries)."""
    path = Path(path)
    if resume and path.exists():
        return json.loads(path.read_text())
    w, tries = match_w_scale_batch(brain, w_ref, target, states, others_local)
    m = dict(w_scale=w, frac_firing=activity_target(brain, w, states, others_local), target=target,
             tries=tries)
    path.write_text(json.dumps(m))
    return m


def load_week10(week10_dir):
    """{'connectome, C4': result, 'random network, C4': result} from week 10's output (those present)."""
    d = Path(week10_dir)
    out = {}
    for name, f in (('connectome, C4', 'week10_connectome__c4.pkl'),
                    ('random network, C4', 'week10_random_network__c4.pkl')):
        if (d / f).exists():
            out[name] = pickle.load(open(d / f, 'rb'))
    return out


def compare10b(conn, rands, alpha=0.05):
    """conn: the connectome's C4 result. rands: {name: random network C4 result} (all shuffles, seed 0
    included), all on the same test seeds. Rules fixed before the run:
      * each random network vs the connectome: week 7's verdict (paired by seed, success -> zone ->
        lifted, first exact McNemar p < alpha decides) and placement/time if both >= 10 successes;
      * FINAL: 'the wiring helps' if the connectome is significantly better than a majority of the
        random networks; 'the wiring hurts' if a majority is significantly better than the connectome;
        otherwise 'no effect of the wiring shown'. Placement is reported the same way (count of
        networks significantly more / less precise than the connectome), as a secondary result.
      * every network is also scored against the pass mark (>= 40/50)."""
    rows, better, worse, prec_r, prec_c = {}, 0, 0, 0, 0
    for nm, r in rands.items():
        v, pm = verdict(conn['outs'], r['outs'], alpha)
        sec = secondary(conn['outs'], r['outs'])
        sig = v.split(' is better')[0] if ' is better' in v else None
        better += sig == 'connectome'
        worse += sig == 'random network'
        if sec['compared'] and sec['placement_mm']['p'] < alpha:
            if sec['placement_mm']['b'] < sec['placement_mm']['a']:
                prec_r += 1
            else:
                prec_c += 1
        s = r['summary']
        rows[nm] = dict(success=s['success'], n=s['n'], passed=s['success'] >= PASS * s['n'] / 50,
                        place_mm=s['place_mm_success'], time_s=s['time_s'], w_scale=r.get('w_scale'),
                        verdict=v, success_p=pm['success']['p'], only_conn=pm['success']['only_a'],
                        only_rand=pm['success']['only_b'], secondary=sec, outcomes=s['outcomes'])
    n = len(rands)
    if better > n / 2:
        final = 'the wiring helps'
    elif worse > n / 2:
        final = 'the wiring hurts'
    else:
        final = 'no effect of the wiring shown'
    succ = [rows[k]['success'] for k in rows]
    cs = conn['summary']
    return dict(networks=rows, n_random=n, conn_better=int(better), rand_better=int(worse), final=final,
                rand_more_precise=int(prec_r), conn_more_precise=int(prec_c),
                random_passed=int(sum(r['passed'] for r in rows.values())),
                random_success=dict(min=int(min(succ)), median=float(np.median(succ)), max=int(max(succ))) if succ else None,
                connectome=dict(success=cs['success'], n=cs['n'], place_mm=cs['place_mm_success'],
                                passed=cs['success'] >= PASS * cs['n'] / 50,
                                rank=int(1 + sum(x > cs['success'] for x in succ))))


def describe10b(c):
    L = [f"CONNECTOME, C4: {c['connectome']['success']}/{c['connectome']['n']}, placement "
         f"{c['connectome']['place_mm']} mm; pass: {'YES' if c['connectome']['passed'] else 'NO'}"]
    for nm, r in c['networks'].items():
        sec = r['secondary']
        pl = (f"placement {sec['placement_mm']['a']} vs {sec['placement_mm']['b']} mm (p = {sec['placement_mm']['p']})"
              if sec['compared'] else f"placement not compared ({sec['why']})")
        L.append(f"{nm} (w_scale {r['w_scale']:.3f}): {r['success']}/{r['n']}, pass {'YES' if r['passed'] else 'NO'}; "
                 f"success vs connectome: {r['only_conn']} seeds only the connectome, {r['only_rand']} only this "
                 f"network (p = {r['success_p']:.3g}); {pl}")
        L.append(f"   verdict: {r['verdict']}")
    rs = c['random_success']
    L.append(f"RANDOM NETWORKS: {c['random_passed']}/{c['n_random']} pass; success {rs['min']}-{rs['max']} "
             f"(median {rs['median']:g}); the connectome ranks {c['connectome']['rank']} of {c['n_random'] + 1}")
    L.append(f"FINAL (wiring): {c['final'].upper()}: connectome significantly better than {c['conn_better']} of "
             f"{c['n_random']} random networks, worse than {c['rand_better']}. Placement: {c['rand_more_precise']} "
             f"random networks significantly more precise, {c['conn_more_precise']} less.")
    return L


def restrict(res, seeds):
    """A copy of a run result with only the test episodes in `seeds` (summary recomputed): used to
    pair week 10's 50-seed connectome run with a smoke test's 10 seeds."""
    from .pickplace import summarise_pp
    seeds = set(seeds)
    outs = [o for o in res['outs'] if o['seed'] in seeds]
    return dict(res, outs=outs, summary=summarise_pp(outs))
