# Archive

Code and notebooks from weeks 1–9, kept as a record of how the project got to the final pipeline.
Nothing here is imported by `sim/`, `arm/`, the tests or the final notebooks, and it is not
maintained. Notebook outputs are cleared. The paths inside are the original laptop/Colab/Kaggle
layout (`~/arm_simulation/brain/{sim,data}`, `arm/`), so these won't run from this repo as they are.

- `notebooks/01_status.ipynb`: week 1, neuPrint exploration and the DN shortlist
  (ported to `scripts/fetch_connectome.py`)
- `notebooks/02_weights*.ipynb`: week 2, the signed weight set (VERSION2 was used; ported to
  `scripts/fetch_connectome.py`)
- `notebooks/03_simulation.ipynb`, `04_fullnet_colab.ipynb`: week 2, Brian2 checks and the w_scale choice
- `notebooks/05_week3_speed.ipynb`, `sim/bench.py`, `sim/lif_torch.py`: week 3, engines and benchmark
- `arm/`: week 4, the arm on its own: expert evaluation, imitation baselines, figures
- `notebooks/06_week5_open_loop.ipynb`, `sim/video.py`: week 5, open-loop encoding and decoding
- `notebooks/07_week6_closed_loop.ipynb`: week 6, closed-loop reaching
- `notebooks/08`–`11`: weeks 7–9, pick-and-place, the release fixes and the readout recipes

Results from these weeks are in `../results/`, and the write-ups from the time in `../docs/weekly/`.
