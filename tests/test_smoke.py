"""Smoke tests: CPU only, no connectome data needed (~1-2 min).

  1. every module of the final pipeline imports;
  2. on a small random LIF network, the numba engine and the torch engine on the CPU produce
     identical spikes in float64 (the project's engine check, batch7.py docstring / week 3);
  3. the scripted expert used for the final runs (fix B, sim.expert8) completes pick-and-place in
     the MuJoCo Panda environment on 3 of the test seeds.

The Panda model must be fetched first: scripts/fetch_panda_model.sh
"""
import importlib

import numpy as np
import pytest

MODULES = ['sim.config', 'sim.lif', 'sim.lif_numba', 'sim.batch7', 'sim.network7', 'sim.interface',
           'sim.readout', 'sim.closed_loop', 'sim.pickplace', 'sim.expert8', 'sim.expert9', 'sim.pool8',
           'sim.week7', 'sim.week8', 'sim.week9', 'sim.week10', 'sim.week10b', 'sim.video7',
           'panda_env', 'expert']


@pytest.mark.parametrize('name', MODULES)
def test_imports(name):
    importlib.import_module(name)


def _tiny_network(n=1000, k_out=20, n_stim=40, seed=0):
    """A random signed network in the WeightSet.subnetwork format: integer synapse counts, sign
    set by the presynaptic neuron (80% excitatory), as in the connectome."""
    rng = np.random.default_rng(seed)
    pre = np.repeat(np.arange(n), k_out)
    post = rng.integers(0, n, len(pre))
    sign = np.where(rng.random(n) < 0.8, 1, -1)
    w = rng.integers(1, 30, len(pre)) * sign[pre]
    sub = dict(keep=np.arange(n), pre=pre.astype(np.int32), post=post.astype(np.int32), w=w.astype(np.float64))
    return sub, np.arange(n_stim)


def test_numba_torch_cpu_identical_spikes():
    torch = pytest.importorskip('torch')
    from sim.batch7 import NumbaBatchLIF, TorchBatchLIF
    sub, stim = _tiny_network()
    n = len(sub['keep'])
    seeds, w_scales = [11, 12, 13], [1.0, 1.5, 2.0]
    prob = np.full((len(seeds), len(stim)), 150.0 * 0.1e-3)          # 150 Hz input, dt = 0.1 ms
    rec = np.arange(n)
    n_steps, bin_steps = 2000, 50                                    # 200 ms, 5 ms bins

    num = NumbaBatchLIF(sub, stim, verbose=False)
    num.reset(seeds, w_scales, track=True)
    c_num = num.advance(prob, n_steps, rec, bin_steps)

    tor = TorchBatchLIF(sub, stim, device='cpu', dtype='float64', verbose=False)
    tor.reset(seeds, w_scales, track=True)
    c_tor = tor.advance(prob, n_steps, rec, bin_steps)

    others = np.setdiff1d(rec, stim)
    assert c_num[:, :, others].sum() > 100, "the test network should be active beyond its inputs"
    assert np.array_equal(c_num, c_tor), "numba and torch-CPU (float64) spike counts differ"
    assert np.array_equal(num.totals(), tor.totals().astype(np.int64))


@pytest.mark.parametrize('seed', [93000, 93001, 93002])
def test_scripted_expert_places_the_cube(seed):
    from panda_env import PandaPickPlace, OBS_NAMES
    from sim.pickplace import ExpertPolicy, run_pp
    from sim.week9 import EXPERTS
    ex = EXPERTS['B']                                                # the teacher of weeks 8-10 ("fix B")
    out = run_pp(PandaPickPlace(), ExpertPolicy(ex, OBS_NAMES), ex, seed)
    assert out['success'], f"seed {seed}: {out['outcome']}"
    assert out['place_error_mm'] < 20
