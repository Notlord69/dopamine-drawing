# A fly connectome as a robot controller: does the wiring matter?

## 1. Question

Connectomes now map every neuron and synapse of an animal's nervous system. Shiu et al. (2024)
showed that a simple spiking model built straight from the fly brain's wiring diagram, with no
fitted parameters per neuron, reproduces some of the fly's own sensorimotor responses. This project
asks two narrower questions:

1. Can a **frozen** whole-CNS connectome model act as the controller of an unrelated body (a robot
   arm doing pick-and-place) if only a linear readout is trained?
2. If it can, does that depend on the fly's **specific wiring**, or would any network with the same
   size, degrees, weights and excitation/inhibition balance do as well?

The answers: yes, 46/50, and no, not as far as this test can tell.

## 2. Setup

**The network.** FlyEM's male *Drosophila* CNS connectome v1.0 (brain and ventral nerve cord):
176,422 neurons and 24,287,791 signed connections. Each connection's weight is its synapse count.
Its sign comes from the presynaptic neuron's predicted transmitter: acetylcholine excites; GABA,
glutamate and histamine inhibit; modulators are left out. Every neuron is the same leaky
integrate-and-fire unit with Shiu et al.'s parameters (time step 0.1 ms, 1.8 ms synaptic delay). One
global synapse strength scales them all: w_scale 0.70, chosen in week 2 by a firing-rate rule on
sugar and JO-C/E stimulation, before any arm was involved. Nothing in the network is trained, and nothing changes during a run.

**The loop** (diagram in the README). Every 40 ms the arm's state is reduced to 8 signals: the
cube's position relative to the gripper (x, y, z), the grip width, the target's position relative
to the cube (x, y), the gripper height and the cube height. Each signal drives a push channel and a
pull channel of Poisson spikes (150 ± 125 Hz) into the 335 JO-C/E mechanosensory neurons of the
antenna. The 16 channels are assigned by splitting those neurons on their own output wiring. The
network then runs for 80 ms. The spike counts of 3,000 readout neurons over those 80 ms go through
a linear (ridge) map to four numbers: dx, dy, dz (≤ 2 cm per step) and grip (open or close). A
damped least-squares IK controller turns those into joint commands for a simulated Franka Panda in
MuJoCo.

**Training.** Only the readout is fitted. A memoryless scripted expert labels every state the arm
visits. DAgger runs 4 rounds of 30 episodes (training seeds 64000 and up). In round 0 the expert
drives; after that the readout drives. In half the episodes 5 mm of action noise is added, so the
data covers recoveries. The 3,000 readout neurons are re-chosen at every refit: the neurons whose
rates correlate best with the expert's actions, from a pool of all neurons active during an expert
episode (≥ 1 Hz) plus all descending neurons.

**The task and the test.** A 5 cm cube appears at a random spot on the table (fixed yaw). It has to
be picked up, carried to a 5 cm target zone, put down, released, and the gripper lifted 6 cm clear.
That state has to hold for 0.4 s, all within 6 s. Success is judged by the physics only. The
**pass mark, ≥ 40/50, was fixed in week 7**, before any of these results. The final test uses
**fresh seeds 93000–93049**, never used before week 10.

**The control network.** A degree-preserving shuffle: every connection keeps its source, synapse
count and sign, but its target is permuted. So every neuron keeps its exact in- and out-degree, and
the excitation/inhibition balance is unchanged. Only *who connects to whom* is destroyed. Each
shuffle gets its own w_scale, set so that the same share of non-input neurons fires at ≥ 1 Hz as in
the connectome (4.5%). It also gets its own encoder channels, its own active-neuron pool and its own
readout, all by the same rules.

## 3. Results

Fresh seeds 93000–93049. DAgger is the same for every policy (4 × 30). Kaggle Tesla T4, float32.

| policy | success (95% CI) | placement, successes (median) | time (median) |
|---|---|---|---|
| expert (scripted teacher, fix B) | 50/50 (93–100%) | 2.6 mm | 1.84 s |
| **connectome + linear readout (C4)** | **46/50 (81–97%)** | 14.3 mm | 3.08 s |
| random network, seed 0 (w_scale 1.133) | 50/50 (93–100%) | 5.4 mm | 3.26 s |
| random network, seed 1 (1.057) | 49/50 (90–100%) | 10.8 mm | 3.24 s |
| random network, seed 2 (1.082) | 45/50 (79–96%) | 9.4 mm | 3.40 s |
| random network, seed 3 (1.082) | 50/50 (93–100%) | 6.0 mm | 3.12 s |

The connectome passes. So do all four random networks, 194/200 between them.

**The wiring test** was pre-registered in week 7. It pairs the two networks by seed and checks the
milestones in turn (success, then carried to the zone, then lifted) with an exact McNemar test at
p < 0.05. Final rule: the wiring helps if the connectome is significantly better than a majority of
the random networks. On success:

| random network | only the connectome succeeds / only the random network | p |
|---|---|---|
| seed 0 | 0 / 4 | 0.125 |
| seed 1 | 1 / 4 | 0.375 |
| seed 2 | 5 / 4 | 1 |
| seed 3 | 0 / 4 | 0.125 |

The connectome is significantly better than 0 of 4 and worse than 0 of 4: **no effect of the wiring
shown**. On placement (Mann–Whitney on successes), 3 of the 4 random networks are significantly
*more* precise than the connectome (p < 0.0001, 0.022, < 0.0001), and the fourth points the same
way (p = 0.051). The 10 failures across all five networks fall on 10 different seeds, so no seed is
simply hard.

## 4. Controls and ablations

Each result below is on 50 test seeds unless noted. Every no-brain control was trained exactly like
the brain (same DAgger, seeds and noise) on the 8 noise-free signals.

- **No network, linear readout on the 8 signals: 0/50**, plain or with C4's row weights. The
  network is doing work that a linear map of its inputs can't.
- **No network, MLP on the 8 signals: 50/50**, 2.6 mm, as good as the expert. This is why the
  readout must stay linear. With an MLP readout the network would not be needed at all.
- **Feedback cut: 0/10**, never lifts the cube. The encoder is frozen at the first state, so the
  network gets no arm feedback. The same holds for all five networks (10/10 never lifted each). The
  arm movements come from the closed loop, not from a replayed pattern.
- **DN-only readout: 0/50** (week 8). Reading out only the 1,332 descending neurons, the fly's own
  output channels, carries the cube but never finishes.
- **Readout width: 1,332 → 3,000 neurons, 0 → 25/50** (week 8b). The failures moved to the retreat
  after release.
- **Release-region weighting (C4): 25 → 46/50** (week 9). The rows where the cube sits on the target
  carry a third of the fit.
- **Press weighting (C6): 0/50.** Adding up-weighted "press" rows (holding the cube just above the
  target) taught the readout to go down whenever it was holding. It lifted every cube and placed none.
- **Grip-gated readout (G): ruled out.** Two linear motion maps switched by the decoded grip did
  34/50 on the fresh seeds (41/50 in an earlier CPU run) **with no network at all**. A brain result
  with this readout could not show the brain was needed, so it wasn't run on the brain.
- **Four random networks** (above): no significant difference on success, and 3 of 4 place more
  precisely.

## 5. What it shows, and what it doesn't

The defensible claim:

> *A frozen spiking network with the fly brain's size, degree distribution, synaptic weights and
> E/I balance, read out linearly from 3,000 active neurons, can do pick-and-place.*

Not: *the fly's wiring solves pick-and-place.* The random networks do at least as well. With 50
seeds per network, an effect of a few seeds either way would not be detected. But nothing here
points to the connectome's specific wiring helping, and placement points the other way. The
remaining failure mode, "the press" (hovering above the target still holding the cube), shows up in
both the connectome and the random networks. It belongs to the C4 readout recipe, not to any
wiring.

## 6. Lessons

1. **The bottleneck was the readout, not the brain or the teacher.** Every failure in weeks 7–8b
   was regression to the mean. A linear map with too little information predicts the average action
   wherever states look alike. Near the target, states that need opposite moves (down to place the
   cube, up after letting go) look alike to a narrow readout. It predicts their average, and the
   arm hovers.
2. **Rare phases need their own weight.** The last ~7% of the task needed a third of the fit
   (25 → 46/50). But weighting the wrong rare phase (C6, the press) taught "go down whenever
   holding" and broke the carry.
3. **Teacher changes helped once, then stopped mattering.** The graded teacher in week 8 helped;
   further teacher changes in week 9 didn't.
4. **No-brain controls kept the claims honest.** The MLP and grip-gated readouts were dropped
   because they work without the network.
5. **Pre-registered rules and offline checks.** Every closed-loop decision had a rule written before
   the run, and the final test used fresh seeds. C6 was the one change made without an offline check
   first, and it is the one that failed.

## 7. Limits and open directions

**Limits.**

- One connectome model: uniform LIF parameters, synapse counts as weights, no plasticity, frozen.
- The readout uses central neurons, not only the descending neurons; the DN-only readout failed (0/50).
- The linear readout imitates a scripted expert. The network itself doesn't learn.
- Only degree-preserving shuffles were tested, and only four.
- The task is simulation only, with the cube's yaw fixed at 0 and a 5 cm zone.
- GPU float32 runs aren't bit-reproducible, though their rates are.
- Placement is 3–5× worse than the expert for every network.

**Open directions.**

- A **DN-only readout comparison**. Week 5 found that the arm state could be decoded from the
  connectome's DN candidates (median R² 0.84) but not from a shuffled network's (0.28). That is
  where the wiring should matter, but a DN-only readout would first have to pass the task.
- **Other null models**, for example shuffles that keep the cell-type block structure, so the test
  asks about finer wiring than "anything with the same degrees".

## 8. Reproducing

- **Hardware.** The final runs used a Kaggle notebook with a Tesla T4. Week 10 takes ~100 min,
  including the no-brain controls on the CPU and a warm start from week 9's DAgger data; without
  that it is ~40 min more. Week 10b takes ~110 min.
- **Engines.** The numba CPU engine runs the full network at ~4.6 s per simulated second on a
  laptop: fine for checks, too slow for the hundreds of episodes in a run. The torch engine batches up
  to 64 episodes on the GPU.
- **Commands.** The README has them: fetch the data (`scripts/fetch_connectome.py`, needs a free
  neuPrint token) and the Panda model (`scripts/fetch_panda_model.sh`), then run
  `notebooks/12_week10_final.ipynb` and `notebooks/13_week10b_shuffles.ipynb`, or
  `scripts/run_final.sh`.
- **Exact numbers.** The results in this write-up are in `results/week10/` and `results/week10b/`.
  A rerun on a GPU will give close but not identical counts, because float32 spikes differ between
  runs.
