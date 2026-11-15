# Data, models and software credits

## Connectome: FlyEM male CNS v1.0 (CC-BY 4.0)

This project uses the **male *Drosophila* central nervous system connectome, v1.0**, by
FlyEM (HHMI Janelia Research Campus) in collaboration with the University of Cambridge (Department of
Zoology), the MRC Laboratory of Molecular Biology and Google Research.

- Website and downloads: https://male-cns.janelia.org (bulk export: https://male-cns.janelia.org/download/)
- neuPrint dataset: `male-cns:v1.0` at https://neuprint.janelia.org
- Licence: [Creative Commons Attribution 4.0 International (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
  The download page says: "The Male CNS dataset is licensed under CC-BY."

**Citation.** The site doesn't give citation text. It links a bioRxiv preprint and a paper in
*Cell*. The preprint, as registered with Crossref:

> Berg S, Beckett IR, Costa M, Schlegel P, Januszewski M, *et al.* (110 authors). Sexual dimorphism
> in the complete connectome of the *Drosophila* male central nervous system. *bioRxiv* (2025),
> posted 9 October 2025. doi:[10.1101/2025.10.09.680999](https://doi.org/10.1101/2025.10.09.680999)

The published version is linked from the site at https://www.cell.com/cell/fulltext/S0092-8674(26)00942-6.

**No connectome data is redistributed in this repository.** `scripts/fetch_connectome.py`
downloads it from the official sources.

**What was changed** (`scripts/fetch_connectome.py`, `sim/lif.py`, `sim/network7.py`):

- Only neuron-to-neuron connections among the 176,422 neurons in neuPrint's `Neuron` table are
  kept, from `connectome-weights-male-cns-v1.0-minconf-0.5.feather`. No synapse-count cutoff is
  applied: 25,862,574 connections and 125,024,863 synapses.
- Each connection's **weight is its synapse count**, signed by the presynaptic neuron's
  **predicted neurotransmitter** (neuPrint's `predictedNt`). Acetylcholine is +1; GABA, glutamate
  and histamine are −1. Dopamine, serotonin, octopamine and tyramine are 0 (the connection is left
  out), leaving 24,287,791 signed connections.
- Transmitters that are missing or unclear (25,159 neurons) are filled with the majority
  transmitter of the same cell type, or acetylcholine if there is none. Histamine fills are allowed
  only in the optic lobe.
- Every neuron gets the **same leaky integrate-and-fire parameters** (Shiu et al. 2024, below). One
  global synapse strength scales all weights (w_scale 0.70). There is no plasticity: the network is
  frozen.
- The control networks are **degree-preserving shuffles** of this weight set: connection targets
  are permuted.
- 335 JO-C/E neurons are used as inputs, driven by Poisson spike trains that encode a robot arm's
  state. This is not their biological input.

These are modelling choices for a robotics experiment. The model is not a validated description of
the fly's nervous system. **No endorsement** of this project by FlyEM, HHMI Janelia, the
collaborating institutions or the dataset's authors is implied.

## Neuron model: Shiu et al. 2024

The LIF neuron and synapse model and its parameters follow:

> Shiu PK, Sterne GR, Spiller N, Franconville R, Sandoval A, Zhou J, Simha N, Kang CH, Yu S, Kim JS,
> Dorkenwald S, Matsliah A, Schlegel P, Yu S, McKellar CE, Sterling A, Costa M, Eichler K, Bates AS,
> Eckstein N, Funke J, Jefferis GSXE, Murthy M, Bidaye SS, Hampel S, Seeds AM, Scott K.
> A *Drosophila* computational brain model reveals sensorimotor processing.
> *Nature* **634**, 210–219 (2024). doi:[10.1038/s41586-024-07763-9](https://doi.org/10.1038/s41586-024-07763-9)

The reference code is https://github.com/philshiu/Drosophila_brain_model (`model.py`). This
project re-implements the same equations, with the changes listed at the top of `sim/lif.py`. No
code from that repository is included.

## Robot model: MuJoCo Menagerie, Franka Emika Panda (Apache-2.0)

- MuJoCo Menagerie `franka_emika_panda`: https://github.com/google-deepmind/mujoco_menagerie.
  Its own `LICENSE` file is the **Apache License 2.0**. The model is derived from Franka Emika's
  `franka_ros` description. It is **not redistributed** here. `scripts/fetch_panda_model.sh` fetches
  it at a pinned commit. `arm/panda_env.py` adds a table, a cube and a target zone at load time and
  scales up the gripper force (10×). It doesn't modify the model files.
- MuJoCo physics engine (Google DeepMind), Apache-2.0: https://github.com/google-deepmind/mujoco

## Software

| package | used for | licence |
|---|---|---|
| Brian2 | reference LIF simulator (weeks 2–3), units | CeCILL-2.1 |
| numba | the CPU engine | BSD |
| PyTorch | the batched GPU engine | BSD-3-Clause (plus bundled components' licences) |
| NumPy, SciPy, pandas | everything | BSD-3-Clause |
| PyArrow | connectome tables | Apache-2.0 |
| MuJoCo (Python bindings) | arm simulation and rendering | Apache-2.0 |
| neuprint-python (HHMI) | neuPrint queries | BSD-style |
| Matplotlib | figures and video panels | Matplotlib licence (PSF-based) |
| imageio, imageio-ffmpeg, Pillow | video | BSD-2-Clause, BSD-2-Clause, MIT-CMU (HPND) |
| FFmpeg | demo video (command-line tool, not bundled) | LGPL/GPL |

## Compute

The final experiments ran on Kaggle notebooks (NVIDIA Tesla T4) and the earlier weeks partly on
Google Colab.
