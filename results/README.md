# Results files

The small outputs of each week's runs, copied unchanged from the project's data folder. The large
ones are left out: DAgger checkpoints (`*_dagger.npz`), run records (`*.pkl`), raw recordings and
anything that lists connectome neuron IDs. The numbers in `README.md` and `WRITEUP.md` come from
`week10/week10_results_kaggle.json` and `week10b/week10b_results_kaggle.json`.

| folder | what | from |
|---|---|---|
| `week2/` | w_scale choice (`wsyn_choice.json`, read by the notebooks), sweep, full-network check, shuffle checks | `archive/notebooks/03–04` |
| `week3/` | engine verification against Brian2, benchmarks, subnetwork fidelity | `05_week3_speed` |
| `week4/` | arm only: expert success and stress tests, imitation baselines | `archive/arm/` |
| `week5/` | open-loop decoding (R² per readout, real vs shuffled) | `06_week5_open_loop` |
| `week6/` | closed-loop reaching, feedback cut | `07_week6_closed_loop` |
| `week7/` | first pick-and-place, connectome vs activity-matched random network (`week7_random_network_meta.json` holds the random network's w_scale, read by the notebooks) | `08_week7_pick_place` |
| `week8/`, `week8b/` | graded teacher; wider readout | `09`, `10` |
| `week9/` | readout recipes C0–C5 offline, C4 closed loop | `11_week9_retreat` |
| `week10/` | **final**: fresh seeds, connectome C4, random network C4, C6, no-brain controls | `notebooks/12_week10_final` |
| `week10b/` | **final**: three more random networks, activity matches, wiring verdict | `notebooks/13_week10b_shuffles` |
| `figures/` | summary plots from weeks 2, 4, 5, 6 and 8 | |

`*_kaggle.json` and `*_colab.json` were written on those hosts. The `warm_from_week9` path in
`week10_results_kaggle.json` is the Kaggle session's temporary folder.
