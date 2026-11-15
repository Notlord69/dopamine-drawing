#!/usr/bin/env python
"""Render demo clips that the original runs did not record, from week 10's trained readouts.

    python scripts/render_demo_episode.py cut   [--net connectome] [--seeds 93000]
    python scripts/render_demo_episode.py press [--net connectome] [--seeds 93035 93039]
    python scripts/render_demo_episode.py test  [--net random]     [--seeds 93000]

  cut    feedback cut: the encoder is frozen at the first state, so the network gets no arm feedback
  press  re-runs test seeds on which week 10's connectome C4 run ended hovering with the cube
  test   any test seed(s), normal closed loop

Needs a GPU machine (Kaggle T4: ~1 min per episode after ~3 min of setup), the data folder from
scripts/fetch_connectome.py and week 10's output (data/week10/week10_*__c4.pkl). Nothing is
retrained: the readout (its 3,000 neurons and weights) is week 10's saved one.

Output in <data>/demo/: <name>_episodes.pkl and <name>.mp4 (sim/video7.py, the renderer the
notebooks used). The script prints each episode's outcome next to week 10's recorded outcome for
that seed. On a GPU in float32 the spikes are not bit-identical between runs (rates are; see
sim/batch7.py), so a re-run can end differently from the original. Caption a clip with the outcome
of the run you show, not the one in the results file.
"""
import argparse, gc, json, os, pickle, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / 'arm')]

import numpy as np

from sim.config import DATA_DIR

NETS = {'connectome': ('week10_connectome__c4.pkl', None),
        'random': ('week10_random_network__c4.pkl', 0)}          # random_network(full, seed=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=('cut', 'press', 'test'))
    ap.add_argument('--net', choices=tuple(NETS), default='connectome')
    ap.add_argument('--seeds', type=int, nargs='*')
    ap.add_argument('--data-dir', type=Path, default=DATA_DIR)
    ap.add_argument('--cpu', action='store_true', help='numba on the CPU (slow: ~5 s per brain-second)')
    ap.add_argument('--slow', type=float, default=0.5, help='video speed (0.5 = half speed, as the notebooks)')
    a = ap.parse_args()
    seeds = a.seeds or {'cut': [93000], 'press': [93035, 93039], 'test': [93000]}[a.mode]

    from sim.lif import WeightSet
    from sim.network7 import random_network
    from sim.pickplace import EnvPool, evaluate_pp
    from sim.week9 import EXPERTS, Week9BrainPolicy, _build_brain
    from sim.week10 import _raster
    from sim.video7 import render
    from panda_env import PandaPickPlace, OBS_NAMES

    data = a.data_dir
    pkl, shuffle_seed = NETS[a.net]
    res = pickle.load(open(data / 'week10' / pkl, 'rb'))
    recorded = {o['seed']: o['outcome'] for o in res['outs']}
    if a.mode == 'cut' and 'cut' in res:
        recorded = {o['seed']: o['outcome'] for o in res['cut']['outs']}

    ws = WeightSet(data / 'weights_male-cns_v1.0')
    jo = ws.by_type(r'^JO-(C|E)', cls='cb_sensory')
    sub = ws.subnetwork(np.arange(ws.N))
    ws._out = None
    gc.collect()
    if shuffle_seed is not None:
        sub = random_network(sub, seed=shuffle_seed)

    cfg = dict(use_gpu=not a.cpu, dtype='float32')
    eng, brain = _build_brain(sub, jo, np.asarray(res['record']), cfg)
    ro = res['readout']
    brain_ms, window_ms = res.get('timing', [80, 80])
    pol = Week9BrainPolicy(brain, res['w_scale'], OBS_NAMES, select='all', weight=True,
                           pool=np.asarray(res['pool_cols']), k=len(ro['cols']), brain_ms=brain_ms,
                           window_ms=window_ms, name=res['name'],
                           max_batch=64 if getattr(eng, 'kind', '') == 'torch' else 1)
    pol.cols, pol.readout = np.asarray(ro['cols']), ro['model']        # week 10's readout, not refitted
    pol.raster_cols = _raster(pol.cols, pol.readout)
    pol.feedback = a.mode != 'cut'

    label = res['name'] + (', feedback cut' if a.mode == 'cut' else '')
    outs, s = evaluate_pp(EnvPool(PandaPickPlace()), pol, EXPERTS['B'], seeds, label=label, record=True)
    eps = []
    for o in outs:
        print(f"  seed {o['seed']}: {o['outcome']}  (week 10 recorded: {recorded.get(o['seed'], 'not run')})")
        eps.append(dict(rec=o['rec'], seed=o['seed'], success=o['success'], outcome=o['outcome'], net=label))

    out = data / 'demo'
    out.mkdir(parents=True, exist_ok=True)
    name = f"{a.net}_{a.mode}_{'_'.join(map(str, seeds))}"
    pickle.dump(eps, open(out / f'{name}_episodes.pkl', 'wb'))
    (out / f'{name}.json').write_text(json.dumps(
        [dict(seed=o['seed'], outcome=o['outcome'], recorded=recorded.get(o['seed'])) for o in outs], indent=1))
    del pol, brain, eng
    gc.collect()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    render(str(out / f'{name}_episodes.pkl'), str(out / f'{name}.mp4'), slow=a.slow)


if __name__ == '__main__':
    main()
