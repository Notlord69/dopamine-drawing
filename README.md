# Dopamine Drawing

**A fly brain learns to draw.** A spiking model of a MaleCNS *Drosophila* subnetwork senses which
way a target dot lies, tries a direction, and gets a dose of reward ("coca cola", the fly's
dopamine) when the pen gets closer. Reward-modulated STDP changes its existing synapses until it
can steer a pen held by a simulated Franka Panda: dots first, then strokes, then "HAPPY BIRTHDAY".

The direction, design and week-by-week plan are in **[`PLAN.md`](PLAN.md)**.

Spun off from [fly-connectome-arm](https://github.com/Notlord69/fly-connectome-arm), where a
frozen whole-CNS model drove the same arm with a trained linear readout. This project keeps that
project's connectome pipeline, LIF engine and arm controller, and moves the learning into the
brain.

## Setup (Python 3.10, CPU)

```bash
python3 -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements.txt && pip install -r requirements-gpu.txt --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -e .
```

```bash
scripts/fetch_panda_model.sh
```

```bash
pytest
```

**Connectome data.** If you already have `weights_male-cns_v1.0/` from fly-connectome-arm, point
`DD_DATA_DIR` at the folder that holds it:

```bash
export DD_DATA_DIR=~/arm_simulation/brain/data
```

Otherwise build it: `export NEUPRINT_TOKEN=...` (free, from neuprint.janelia.org → Account → Auth
token; never commit it), then run `python scripts/fetch_connectome.py`. This is a 1.1 GB download.

## Layout

```
PLAN.md                  master plan: core design, timeline, open decisions, status, decision log
dopamine_drawing/
  config.py              DATA_DIR / MENAGERIE_DIR (env: DD_DATA_DIR, DD_MENAGERIE)
  connectome.py          WeightSet: signed MaleCNS weights, hop/path carving, subnetworks, shuffles
  lif.py                 Shiu et al. 2024 LIF parameters and a batched torch engine (frozen weights for now)
  panda_env.py           MuJoCo Panda + damped least-squares IK (pick-and-place scene; pen env in Week 3)
scripts/                 fetch_connectome.py, fetch_panda_model.sh
tests/                   smoke tests (CPU, no connectome needed)
```

## Data and credits

- **Connectome:** FlyEM male CNS v1.0 (HHMI Janelia FlyEM with the University of Cambridge, the
  MRC Laboratory of Molecular Biology and Google Research), [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  https://male-cns.janelia.org. It is not redistributed here. No endorsement is implied.
- **Neuron model:** Shiu et al. (2024), *Nature* 634, 210–219.
- **Robot:** MuJoCo and the MuJoCo Menagerie Franka Emika Panda model (Apache-2.0), fetched by script.
- Details and the modelling changes: [`ATTRIBUTION.md`](ATTRIBUTION.md). Citation: [`CITATION.cff`](CITATION.cff).

## Licence

Code: [MIT](LICENSE). The connectome data (CC-BY 4.0) and the Panda model (Apache-2.0) keep their
own licences.
