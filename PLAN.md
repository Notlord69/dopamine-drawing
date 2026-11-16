# Dopamine Drawing: master plan

**Goal.** A spiking model of a real fly brain (a MaleCNS subnetwork) learns, through reward alone, to
steer a pen to a target. Then it learns to follow strokes, and finally draws text and pictures with
a simulated Franka Panda: "HAPPY BIRTHDAY", then a cat outline. The reward is called **coca cola**
here. It plays the role dopamine plays in the fly. The fly is never told which way to move. It
tries things, gets coca cola when the pen gets closer, and its synapses change.

This file holds the project's direction. Update the **Status** and **Decision log** sections as
work goes on. Change the core design only through a dated entry in the decision log.

Spun off on 2026-11-16 from *fly-connectome-arm* (`~/fly-connectome-arm`,
github.com/Notlord69/fly-connectome-arm). There a frozen whole-CNS model drove the Panda through
pick-and-place with a linear readout trained by imitation. This project swaps the trained readout
for a **fixed** readout and moves the learning **into the brain's own synapses**.

---

## 1. Core design (decide in Weeks 1–2, then keep fixed)

| piece | design |
|---|---|
| **Sensation** | Target direction ("dot is up-left") → one of **8 groups of input neurons** fires. |
| **Brain** | MaleCNS subnetwork: input neurons → 2–3 hops of interneurons → descending neurons. Aim for **5k–20k neurons**. |
| **Fixed translator** | Descending neurons split into **8 fixed pools**, one per direction. The strongest pool wins and the pen moves that way. **Never trained.** |
| **Learning rule** | **Reward-modulated STDP.** Each synapse keeps an eligibility trace ("I recently helped cause a spike"). When coca cola arrives, traced synapses strengthen or weaken. |
| **Coca cola** | Given when the pen gets closer to the target. Scaled by **surprise** (reward minus the recent average reward), which keeps learning stable. |
| **Constraints** | Only existing synapses change. Signs stay fixed. Weights stay within bounds. Only a chosen subset of synapses is plastic. |
| **Exploration** | Background noise spikes, so the fly "tries things" early on. |

### The rule in equations (starting point for Week 1)

For a plastic synapse i→j with weight w ≥ 0 and fixed sign s (the effective weight is s·w):

```
pre trace      x_i ← x_i·exp(−dt/τ+) + spike_i
post trace     y_j ← y_j·exp(−dt/τ−) + spike_j
STDP kernel    k_ij = A+·x_i·spike_j − A−·y_j·spike_i
eligibility    e_ij ← e_ij·exp(−dt/τ_e) + k_ij           τ_e ≈ 0.2–1 s
coca cola      r_t = closer(pen, target) at each control step
surprise       δ_t = r_t − r̄,   r̄ ← (1−α)·r̄ + α·r_t
update         w_ij ← clip(w_ij + η·δ_t·e_ij, w_min, w_max)   only for plastic synapses
```

Start with τ+ = τ− = 20 ms, w_min = 0, and w_max = a multiple of the connectome synapse count
(say 2–3×). Hold the sign outside w, so a weight can shrink to zero but never flip.

### Interfaces to keep fixed across environments

- **Brain:** `brain.step(input_rates, n_ms) → pool_counts[8]`. Plasticity is on or off, with
  coca cola delivered as `brain.reward(δ)`.
- **Translator:** `direction = argmax(pool_counts)` (ties broken at random) → unit vector × step size.
- **Environment** (point env and Panda share it): `reset(seed) → obs`;
  `step(velocity_xy, pen_down) → obs, info`, where `obs` holds pen xy, target xy and the
  direction bin; `info` holds the distance to the target and the ink.
- **Coca cola** comes from the training loop (distance before vs. after the step), not from the
  environment. Swapping the point env for the arm then changes nothing else.

---

## 2. Timeline

| week | phase | work | checkpoint |
|---|---|---|---|
| 1 | Foundations | R-STDP in PyTorch on a toy network (~100 neurons, 2 inputs, 2 outputs) | learning curve well above chance |
| 2 | Foundations | carve the fly subnetwork; choose the plastic synapses; define the 8 pools | frozen subnet: no silent or exploding regions |
| 3 | Foundations | 2D point env + MuJoCo pen env + classical drawing controller | classical controller draws the card |
| 4 | Learning to move | 1D: left vs. right, 2 actions; tune η, τ_e and the baseline | clear improvement over trials |
| 5 | Learning to move | 2D reaching, 4 → 8 directions; weight bounds and homeostasis | ≥ 80% of dots reached within N steps |
| 6 | Learning to move | move to the Franka: pools → pen velocity → IK; continue from the Week 5 brain | arm reaches dots on paper |
| 7 | Learning to move | **slack**; extras: connectome vs. shuffled, weight drift, before/after spikes | — |
| 8 | Learning to draw | strokes as dot sequences; the planner controls pen up/down; squares, circles, zigzags | recognisable shapes |
| 9 | Learning to draw | Hershey fonts; image → edges → contours → strokes; letters, "HAPPY BIRTHDAY", cat | OCR reads ≥ 70% of characters |
| 10 | Learning to draw | recovery (re-target the nearest waypoint); full logging; gallery | gallery of finished drawings |
| 11 | Demo | website: input, Franka drawing, brain tab, "how the fly learned" replay | live + gallery modes work |
| 12 | Demo | slides, demo video, README, MaleCNS attribution, rehearsal | — |

### Phase 1: Foundations (Weeks 1–3)

**Week 1: the rule on a toy brain.**
- Get the baseline pipeline running again in this repo (env setup, data path, smoke tests).
- Implement R-STDP in PyTorch on a tiny network: ~100 neurons, 2 inputs, 2 outputs.
- Task: when input A fires, reward output 1 (and output 2 for input B).
- **Checkpoint:** a learning curve that rises well above chance. *If the toy can't learn, the fly
  won't either.*

**Week 2: carve out the fly subnetwork.**
- Select the 8 input groups, the descending neurons and the interneurons that connect them
  (2–3 hops, 5k–20k neurons). `connectome.path_subnetwork` already does input → output path
  carving.
- Pick the plastic synapses, for example those onto descending neurons and premotor interneurons.
- Define the 8 fixed direction pools.
- **Sanity check:** the frozen subnet gives reasonable firing rates, with nothing silent or
  exploding. Run it under each input group and under noise alone.

**Week 3: environments.**
- 2D point env: a dot target and a pen point, with the same interface as the arm. Used for fast
  training.
- MuJoCo: a pen on the Panda, paper, ink logging, pen up/down, and a classical drawing controller.
  Start from `panda_env.py` (IK at 500 Hz under a 40 ms policy step).
- **Bonus:** the classical controller can already draw the birthday card, so **the gift does not
  depend on the fly learning in time.**

### Phase 2: Learning to move (Weeks 4–7)

**Week 4: 1D learning (left vs. right).** The target is left or right, and the fly has 2
actions. Tune the learning rate, the trace length (~0.2–1 s) and the reward baseline.
**Checkpoint:** clear improvement over trials.

**Week 5: 2D reaching in the point env.** Go from 4 to 8 directions, reaching random dots. Add
stability measures: weight bounds and firing-rate homeostasis. **Target:** ≥ 80% of dots reached
within N steps. Fix N before the run.

**Week 6: move to the Franka.** The fixed pools drive pen velocity through IK. Learning continues
on the arm from the Week 5 brain. **Checkpoint:** the arm reaches dots on paper.

**Week 7: buffer and analysis.** Slack, because learning projects stall. Extras if on track:
- learning speed: does the real connectome learn faster than a shuffled one?
- weight drift: how far has the brain moved from the original connectome?
- before vs. after: spike patterns at the first trial and the last.

### Phase 3: Learning to draw (Weeks 8–10)

**Week 8: following lines.** A stroke becomes a sequence of dots, and the target is always the
next dot. At first the planner controls pen up/down. Draw squares, circles and zigzags.

**Week 9: text and images.** Text uses Hershey single-stroke fonts. Images go through edges →
contours → simplify → order strokes. Draw letters, then "HAPPY BIRTHDAY", then a cat outline.
**Target:** recognisable, for example OCR reads ≥ 70% of characters.

**Week 10: robustness and logging.** Recovery: if the pen drifts too far, re-target the nearest
waypoint. Log every run: spikes for the displayed neurons, coca cola doses, weights, joint angles
and ink. Build a gallery of finished drawings.

### Phase 4: Demo and presentation (Weeks 11–12)

**Week 11: website.**
- Input: text or an image, with a stroke preview.
- Main screen: the Franka drawing.
- Brain tab: spikes flashing with each action, a coca cola drop animating in on every reward, and
  synapses visibly strengthening.
- "How the fly learned" tab: a replay of the learning curve, early clumsy trials against late
  skilled ones. Likely the biggest wow moment.
- Modes: a precomputed gallery, plus a live mode for short text using the trained brain.

**Week 12: presentation.** Slides: fly → senses target → tries → gets coca cola → learns →
draws. Materials: demo video, README, MaleCNS attribution. Rehearse with the gallery as the
fallback.

---

## 3. Open decisions for Weeks 1–2

Record each choice in the decision log when made.

1. **Which input neurons are the 8 direction groups?** Options: the 335 antennal JO-C/E neurons
   split into 8 by their output wiring (the parent project's split code is in its git history);
   visual projection neurons (more natural for "seeing" a dot); or a mix. Prefer groups whose
   downstream targets overlap little.
2. **Which descending neurons form the 8 pools?** Options: DNs reachable within 2–3 hops of the
   inputs, split by wiring or at random into 8 equal pools. They need enough baseline activity
   that every pool can win sometimes.
3. **Subnetwork size and hop depth.** Target 5k–20k neurons. Watch the connection count, since
   eligibility traces cost memory per plastic synapse.
4. **The plastic subset.** For example, synapses onto DNs plus onto premotor interneurons. Start
   small, so credit assignment stays tractable.
5. **Global synapse scale.** The parent project chose w_scale 0.70 (w_syn = 0.1925 mV) on the full
   network. A small subnetwork has lost most of its input and will likely need a larger scale plus
   background noise. Choose it by a firing-rate rule written down before looking at learning.
6. **Pool readout window** (e.g. 40–80 ms per action) and **step size** of the pen.
7. **Reward shaping.** Distance decrease (graded) or a fixed dose when closer; a bonus on reaching
   the dot; the baseline's averaging rate α.

## 4. Lessons carried over from fly-connectome-arm

- **The wiring may not matter.** There, four degree-preserving shuffles did as well as the
  connectome. For the Week 7 "real vs. shuffled" comparison, fix the test and the pass rule before
  running. Give each shuffle its own activity-matched global scale (match the fraction of neurons
  firing ≥ 1 Hz), or the comparison is unfair.
- **Shuffles can silence the readout neurons.** In that project's week 5, random rewiring dropped
  DN decoding R² from 0.84 to 0.28 before activity matching.
- **A DN-only readout was too poor for pick-and-place** (0/50 in its week 7; it needed 3,000
  central neurons). Here the task needs only 8 directions, but check early (Week 2) that the DN
  pools respond differently to the 8 input groups at all.
- **Sparse activity:** at w_scale 0.70 only ~4.5% of non-input neurons fired ≥ 1 Hz. Synapses that
  never see spikes never become eligible, which is another reason for exploration noise.
- **GPU batching matters:** one episode at a time on the GPU was slower than the laptop CPU, but
  32 side by side cost ~1.2 s each. Keep a batch dimension in the plastic engine from the start
  (each batch member can have its own weights, or the members share weights and average their
  updates).
- **GPU float32 spikes are not bit-reproducible** between runs. Compare rates and success counts,
  not exact spikes.
- **Pre-registering** pass marks and test seeds before runs worked well. Keep doing it.
- The Panda IK (damped least squares on the *commanded* joint angles, gravity and velocity
  feed-forward) is solid at 40 ms policy steps. Reuse it.

## 5. Risks and fallbacks

| risk | fallback |
|---|---|
| R-STDP doesn't learn on the toy | debug there (trace length, reward sign, baseline) before touching the fly |
| the fly subnet doesn't learn 8 directions | 4 directions; a smaller plastic subset; more noise; a stronger reward |
| learning works in the point env but not on the arm | train in the point env, freeze, and run the frozen brain on the arm |
| learning is too slow for drawing | the classical controller draws the card (Week 3); the fly's drawing becomes the "how it learned" story |
| GPU time | toy and point env on CPU; batch episodes |

## 6. Repo layout

```
PLAN.md                  this file
dopamine_drawing/        the package
  config.py              DATA_DIR / MENAGERIE_DIR (env: DD_DATA_DIR, DD_MENAGERIE)
  connectome.py          WeightSet: load the signed weight set, hop/path carving, subnetworks, shuffles
  lif.py                 Shiu et al. LIF parameters; batched torch LIF engine (frozen weights)
  panda_env.py           MuJoCo Panda scene + IK (still the pick-and-place scene; becomes the pen env in Week 3)
scripts/
  fetch_connectome.py    download the connectome and build the weight set
  fetch_panda_model.sh   fetch the Menagerie Panda (pinned)
tests/                   smoke tests
```

Planned additions, in order: `rstdp.py` (Week 1), `subnet.py` + `pools.py` (Week 2), `envs/`
(Week 3: point env, pen env, classical controller), `train.py` (Week 4), `strokes.py` (Weeks 8–9),
`web/` (Week 11). Small results files go in `results/`, large outputs in `outputs/` (git-ignored).

## 7. Status

| date | week | status |
|---|---|---|
| 2026-11-16 | 0 | repo restructured from fly-connectome-arm; master plan written |

## 8. Decision log

| date | decision | why |
|---|---|---|
| 2026-11-16 | Keep the Shiu et al. LIF model, the signed synapse-count weight set and the torch engine from the parent project | known to work on the MaleCNS v1.0 connectome; the plastic engine extends the torch engine |
| 2026-11-16 | Drop Brian2, numba, the ridge readout, the encoders and DAgger | no trained readout here; R-STDP is written in torch |
