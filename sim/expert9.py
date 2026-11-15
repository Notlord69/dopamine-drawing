"""Week 9: fix B2, the graded retreat. A teacher change only, like fix B (expert8).

Week 8b: with the 3,000-neuron readout the brain does everything up to the release (25/50), and all
25 failures are the same: the cube is put down, the gripper opens, and the decoded dz then sits at
about 0 +- 10 mm/step, so the open gripper hovers just above the cube until time runs out. In
expert8 the release is a cliff in the label: while the fingers open (3 steps, width 50 -> 67 mm)
dz is 0, and the next step, once the width passes 70 mm, it is +20 mm/step. Neighbouring states
differ mainly in the gripper width, so a linear readout of noisy rates averages the two.

B2 changes only that branch (cube low and at the target, fingers still around it): the upward
speed grows with how far the fingers have opened,
    f  = clip((width - W_START) / (W_FULL - W_START), 0, 1)
    dz = f * (expert8's retreat dz)
so dz ramps 0 -> +20 mm/step over the opening instead of jumping at 70 mm. W_START is just above
the cube's width (50 mm): the gripper does not rise while the fingers still touch the cube, so it
cannot lift the cube back up. Everything else is expert8 unchanged, and it is still memoryless.
"""
import numpy as np

from expert import move, HOVER_Z, CUBE_HALF, STRADDLE_XY, OPEN
from .expert8 import make_expert8, RELEASE_DZ, RELEASE_XY

W_START = CUBE_HALF * 2 + 0.002     # 52 mm: the fingers have left the cube
W_FULL = CUBE_HALF * 2 + 0.020      # 70 mm: expert8's release/retreat boundary (full speed)

_e8 = make_expert8()


def make_expert9(w_start=W_START, w_full=W_FULL, release_dz=RELEASE_DZ, release_xy=RELEASE_XY):
    def expert9(o):
        a, phase = _e8(o)
        if phase != 'release':
            return a, phase
        tcp, width = o[:3], o[6]
        f = float(np.clip((width - w_start) / (w_full - w_start), 0, 1))
        up = move(tcp, np.r_[tcp[:2], HOVER_Z])[2]
        return np.r_[0, 0, f * up, OPEN], 'release'
    return expert9


expert9 = make_expert9()
