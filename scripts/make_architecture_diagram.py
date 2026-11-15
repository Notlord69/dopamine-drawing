#!/usr/bin/env python
"""Draw the closed loop: docs/architecture.svg and docs/architecture.png (also the demo's diagram card).

    python scripts/make_architecture_diagram.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / 'docs'
INK, MUTED, LINE = '#1d1d1b', '#5b5a56', '#8a8984'
ARM, NET, READ, TRAIN, WARN = '#e8f0fb', '#fbeee6', '#e9f6ef', '#f3f0fb', '#b4441f'

fig = plt.figure(figsize=(12.8, 5.4), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 128); ax.set_ylim(0, 54); ax.axis('off')
fig.patch.set_facecolor('white')


def box(x, y, w, h, title, sub='', fc=ARM, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.25,rounding_size=1.2', fc=fc,
                                ec=LINE, lw=1.2, ls='--' if dashed else '-'))
    nl = sub.count('\n') + 1 if sub else 0
    block = 2.4 + 2.0 * nl                       # title line + subtitle lines, centred in the box
    ty = y + h / 2 + block / 2 - 1.2
    ax.text(x + w / 2, ty, title, ha='center', va='center', fontsize=10.5, weight='bold', color=INK)
    if sub:
        ax.text(x + w / 2, ty - 1.6, sub, ha='center', va='top', fontsize=8.3, color=MUTED,
                linespacing=1.35)
    return (x, y, w, h)


def arrow(p, q, dashed=False, color=LINE, rad=0.0, label=None, lpos=0.5, loff=(0, 1.4)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=13, lw=1.4, color=color,
                                 ls='--' if dashed else '-', connectionstyle=f'arc3,rad={rad}',
                                 shrinkA=2, shrinkB=2))
    if label:
        x = p[0] + (q[0] - p[0]) * lpos + loff[0]; y = p[1] + (q[1] - p[1]) * lpos + loff[1]
        ax.text(x, y, label, ha='center', va='bottom', fontsize=8, color=MUTED)


ax.text(2, 51, 'Closed loop, every 40 ms control step', fontsize=12, weight='bold', color=INK, va='center')

# top row: arm -> signals -> encoder -> input neurons -> network
r1y, h = 33, 11
a = box(2, r1y, 23, h, 'MuJoCo Panda', 'arm + cube state\n40 ms step', ARM)
s = box(29, r1y, 17, h, '8 signals', 'cube − gripper x, y, z;\ngrip; goal − cube; heights')
e = box(49, r1y, 19, h, 'Encoder', '16 push/pull Poisson\nchannels (150 ± 125 Hz)', NET)
j = box(73, r1y, 16, h, '335 JO-C/E', 'mechanosensory\ninput neurons', NET)
n = box(94, r1y - 5, 32, h + 10, 'Frozen LIF network',
        'male CNS v1.0: 176,422 neurons,\n24.3M connections, w_scale 0.70\n80 ms of brain per step\nGPU (torch) / CPU (numba)',
        NET)

# bottom row, right to left: spikes -> readout -> action -> IK -> arm
r2y = 9
k = box(100, r2y, 26, h, 'Spike counts', '3,000 readout neurons,\nlast 80 ms', READ)
rr = box(73, r2y, 22, h, 'Ridge readout (C4)', 'linear map, release\nregion weighted ⅓', READ)
act = box(49, r2y, 19, h, 'dx, dy, dz, grip', '≤ 2 cm per step;\ngrip open / close', READ)
ik = box(2, r2y, 23, h, 'Damped least-squares IK', 'gripper held pointing down,\n500 Hz joint servos', ARM)

mid = lambda b, side: {'r': (b[0] + b[2], b[1] + b[3] / 2), 'l': (b[0], b[1] + b[3] / 2),
                       't': (b[0] + b[2] / 2, b[1] + b[3]), 'b': (b[0] + b[2] / 2, b[1])}[side]
arrow(mid(a, 'r'), mid(s, 'l'))
arrow(mid(s, 'r'), mid(e, 'l'))
arrow(mid(e, 'r'), mid(j, 'l'))
arrow(mid(j, 'r'), (n[0], r1y + h / 2))
arrow((113, n[1]), mid(k, 't'))
arrow(mid(k, 'l'), mid(rr, 'r'))
arrow(mid(rr, 'l'), mid(act, 'r'))
arrow(mid(act, 'l'), mid(ik, 'r'))
arrow(mid(ik, 't'), mid(a, 'b'))

# training path (dashed side branch)
t = box(30, 0.6, 37, 5.4, 'Scripted expert + DAgger (4 rounds × 30 episodes)', '', TRAIN, dashed=True)
ax.texts[-1].set_fontsize(9)
arrow((t[0] + t[2], t[1] + t[3] / 2), (rr[0] + 6, rr[1]), dashed=True, color='#6250d6')
ax.text(73.5, 3.0, 'fits the readout only;\nthe network is never trained', fontsize=7.6, color='#6250d6',
        va='center', ha='left')

# the two controls, as annotations
ax.text(35, r1y - 4.6, '✕ feedback cut (input frozen) → never lifts the cube, 0/10',
        fontsize=8.6, color=WARN, ha='center', weight='bold')
ax.text(60, r2y + h + 2.6, '✕ no network (linear on the 8 signals) → 0/50',
        fontsize=8.6, color=WARN, ha='center', weight='bold')

OUT.mkdir(exist_ok=True)
fig.savefig(OUT / 'architecture.svg', facecolor='white')
fig.savefig(OUT / 'architecture.png', facecolor='white', dpi=100)
print('wrote', OUT / 'architecture.svg', OUT / 'architecture.png')
