"""Smoke tests: CPU only, no connectome data needed (a few seconds).

  1. every module imports;
  2. the torch LIF engine on a small random signed network is deterministic for a fixed seed,
     responds to input, and dropping trials from the batch leaves the kept trials unchanged;
  3. subnetwork carving and the degree-preserving shuffle keep what they should;
  4. the MuJoCo Panda scene loads and the IK moves the gripper where it is told
     (needs scripts/fetch_panda_model.sh; skipped otherwise).
"""
import importlib

import numpy as np
import pytest

MODULES = ['dopamine_drawing.config', 'dopamine_drawing.connectome', 'dopamine_drawing.lif',
           'dopamine_drawing.panda_env']


@pytest.mark.parametrize('name', MODULES)
def test_imports(name):
    importlib.import_module(name)


def _tiny_network(n=1000, k_out=20, n_stim=40, seed=0):
    """A random signed network in the subnetwork format: integer synapse counts, sign set by the
    presynaptic neuron (80% excitatory), as in the connectome."""
    rng = np.random.default_rng(seed)
    pre = np.repeat(np.arange(n), k_out)
    post = rng.integers(0, n, len(pre))
    sign = np.where(rng.random(n) < 0.8, 1, -1)
    w = rng.integers(1, 30, len(pre)) * sign[pre]
    sub = dict(keep=np.arange(n), pre=pre.astype(np.int32), post=post.astype(np.int32), w=w.astype(np.float64))
    return sub, np.arange(n_stim)


def _run(eng, seeds, rate_hz, steps=2000, keep=None):
    n_stim = len(eng.stim_np)
    eng.reset(seeds, [1.0] * len(seeds))
    prob = np.full((len(seeds), n_stim), rate_hz * eng.dt * 1e-3)
    c = eng.advance(prob, steps // 2, np.arange(eng.n), 50)
    if keep is not None:
        eng.keep(keep)
        prob = prob[keep]
    return c, eng.advance(prob, steps // 2, np.arange(eng.n), 50)


def test_lif_engine():
    pytest.importorskip('torch')
    from dopamine_drawing.lif import TorchBatchLIF
    sub, stim = _tiny_network()
    eng = TorchBatchLIF(sub, stim, device='cpu', dtype='float64', verbose=False)
    a1, a2 = _run(eng, [1, 2, 3], 150.0)
    b1, b2 = _run(eng, [1, 2, 3], 150.0)
    assert np.array_equal(a1, b1) and np.array_equal(a2, b2), "same seeds, different spikes"
    assert a1.sum() > 0, "network silent under input"
    _, k2 = _run(eng, [1, 2, 3], 150.0, keep=[0, 2])
    assert np.array_equal(k2, a2[[0, 2]]), "dropping a trial changed the others"
    q1, q2 = _run(eng, [1], 0.0)
    assert q1.sum() + q2.sum() == 0, "spikes without input"


def test_subnetwork_and_shuffle():
    from dopamine_drawing.connectome import shuffle_sub
    sub, _ = _tiny_network(n=200, k_out=5)
    sh = shuffle_sub(sub, seed=1)
    n = len(sub['keep'])
    for side in ('pre', 'post'):
        assert np.array_equal(np.bincount(sub[side], minlength=n), np.bincount(sh[side], minlength=n))
    assert np.array_equal(sub['w'], sh['w']) and not np.array_equal(sub['post'], sh['post'])


def test_panda_ik():
    from dopamine_drawing import panda_env as pe
    if not (pe.MENAGERIE / 'scene.xml').exists():
        pytest.skip('Panda model not fetched (scripts/fetch_panda_model.sh)')
    env = pe.PandaPickPlace()
    env.reset(seed=0)
    x0 = env.tcp_pos()
    for _ in range(25):                                  # 1 s, 10 cm in +x at 4 mm per step
        env.step([0.004, 0.0, 0.0, 1.0])
    for _ in range(10):                                  # settle
        env.step([0.0, 0.0, 0.0, 1.0])
    moved = env.tcp_pos() - x0
    assert abs(moved[0] - 0.10) < 0.01 and np.abs(moved[1:]).max() < 0.01, moved
