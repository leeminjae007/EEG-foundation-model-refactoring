# knn37 local parameter search

Campaign started 2026-09-16. Base campaign: `outputs/knn37_downstream_all13_20260915`.
The pretrained checkpoint and each dataset's head, batch and cosine horizon are retained.

| Dataset | First seed | Common LR candidates | WD candidates | Dropout | Minimum epochs / patience |
|---|---:|---|---|---|---|
| HMC | 1001 | 5e-5, 1e-4, 1.5e-4, 5e-4 | .025, .05, .1 | .1, .2, .3 | 12 / 8 |
| ISRUC | 696 | 5e-5, 1e-4, 1.5e-4, 5e-4 | .005, .01, .02 | .1, .2, .3 | 6 / 5 |
| MentalArithmetic | 3407 | 5e-5, 1e-4, 1.5e-4, 5e-4 | .005, .01, .02 | .1, .2, .3 | 6 / 5 |
| TUEV | 1001 | 5e-5, 1e-4, 1.5e-4, 5e-4 | .005, .01, .02 | .1, .2, .3 | 12 / 8 |
| TUAB | 696 | 5e-6, 1e-5, 1.5e-5 | 2.5e-5, 5e-5, 1e-4 | .1, .2, .3 | 4 / 3 |

All three optimizer groups use the same selected LR from the first update.
Multiclass label smoothing stays at .1. Binary loss retains its existing implementation.
Early stopping applies only to this campaign's source snapshot and does not shorten the cosine horizon.

The initial 34 runs change one axis at a time and reuse the completed, matching baseline.
Only axes with positive validation gains are combined (at most four combinations per dataset).
The top two candidates with a positive first-seed validation gain are evaluated on two additional seeds,
chosen from the existing fixed seed order before observing their candidate scores.
Promotion requires a positive average delta against the matching baseline on all three seeds and on
the two new seeds alone. The winner is locked before training the remaining two seeds and before any
test evaluation. If no candidate qualifies, the existing five-seed baseline is retained explicitly.

Each dataset allows at most two concurrent GPU runs. CPU-only dependent jobs advance stages.
GPU trials request four hours on `gl40s_dev,gl40s_short,gl40s_long`, including later stages.
The wall limit is independent of validation early stopping. A timed-out run without a result
uses the existing single missing-artifact retry and resumes `last.pth` when available.
Missing artifacts are recorded; missing-file failures get one bounded retry using the last checkpoint
where available. Nonfinite gradients mark a trial as pruned, not completed. Other failures remain visible.

Local progress: `outputs/results/search-09161157-knn37-hp/`.
The local monitor publishes final tables only when all five datasets have exactly five complete,
readable final results, preserving per-seed metrics and updating the global results index.

Validation: `python -m pytest ablation/tests/test_local_search.py -q`.
The CPU preflight additionally exercises real train/validation data, full-model initialization,
optimizer updates, base LR, checkpoint writing and test gating for each dataset.
