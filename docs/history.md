# How the project got here, week by week

The short version of the path from a connectome download to the final wiring test. The weekly
write-ups from the time are in [`weekly/`](weekly/) (weeks 1–6) and the notebooks in
[`../archive/notebooks/`](../archive/notebooks/). Their numbers are in [`../results/`](../results/).
Every closed-loop decision from week 6 on had its rule written down before the run.

| week | what was done | result |
|---|---|---|
| 1–2 | Downloaded the FlyEM male CNS v1.0 connectome (neuPrint + the bulk export). Built a signed weight set (synapse counts, signs from predicted transmitters). Ported Shiu et al.'s LIF model to Brian2 and chose the synapse strength. Shortlisted 68 descending-neuron (DN) candidates between the antennal JO-C/E neurons and the front-leg motor neurons. | w_scale 0.70 (w_syn = 0.1925 mV); 335 JO-C/E inputs, 68 DN candidates |
| 3 | A numba CPU engine and a batched torch GPU engine for the full network (176,422 neurons, 24,287,791 connections), checked spike for spike against Brian2. | numba 4.6 s per simulated second on the laptop, identical to Brian2; GPU batching ~1.2 s per episode-second |
| 4 | MuJoCo pick-and-place with the Menagerie Panda: damped least-squares IK and a memoryless scripted expert. No brain yet. | expert 500/500; linear map on the raw state 0%; MLP ~100% |
| 5 | Open loop: arm state encoded into the JO-C/E neurons as push/pull Poisson channels and decoded from the DN candidates. | arm state decodable from the DN candidates (median R² 0.84); a degree-preserving random rewiring silences them (0.84 → 0.28) |
| 6 | Closed-loop reaching with a ridge readout of the 68 DN candidates, trained by DAgger. | 50/50; feedback cut 0/20 |
| 7 | Full pick-and-place, read out from all 1,332 descending neurons, against an activity-matched random network. The pass mark (≥ 40/50) and the wiring test (paired McNemar) were fixed here. | 0/50 for both networks: they carry the cube but can't finish |
| 8 | A graded "press-down" teacher for the release (fix B), still read out from the 1,332 DNs. | still 0/50 (random network too): regression to the mean |
| 8b | Readout widened to the 3,000 most informative active neurons. | 25/50; the failures all in the retreat |
| 9 | Readout neurons re-chosen on all data, and the release region weighted to a third of the fit (readout C4). | 46/50 on the old test seeds |
| 10 | Fresh test seeds 93000–93049. Connectome C4 vs random network (seed 0), C6 (C4 + press weighting), no-brain controls. | connectome 46/50, random 50/50, C6 0/50; no wiring effect |
| 10b | Three more random networks, each activity-matched the same way. | 49, 45, 50 → no effect of the wiring shown (0 of 4 differ) |

What it adds up to is in [`../WRITEUP.md`](../WRITEUP.md).
