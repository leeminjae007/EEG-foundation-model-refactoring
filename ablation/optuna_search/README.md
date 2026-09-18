# Five-seed Optuna finetuning

This isolated campaign replaces the cancelled validation-based local search.
Each trial trains seeds 42, 696, 1001, 1234, and 3407. The objective is the mean
test balanced accuracy of those five runs. Each seed's checkpoint selection and early
stopping use validation balanced accuracy; test is measured only after that choice.
Optuna compares trials using test scores, so these remain test-informed exploratory
results, not an independent held-out final evaluation.

| Dataset | Candidate budget | Common LR | Weight decay | Dropout | Minimum epochs / patience |
|---|---:|---|---|---|---|
| TUEV | 16 | 5e-5, 1e-4, 1.5e-4, 5e-4 | .005, .01, .02, .05, .1 | .1, .2, .3 | 12 / 8 |
| TUAB | 12 | 5e-6, 1e-5, 1.5e-5, 5e-5 | 5e-5, .005, .05, .1 | .1, .2, .3 | 5 / 3 |
| Mental Arithmetic | 1 (trial 0 only) | 1e-4 | .01 | .2 | 5 / 5 |
| ISRUC | 12 | 5e-5, 1e-4, 1.5e-4, 5e-4 | .005, .01, .02, .05, .1 | .1, .2, .3 | 8 / 5 |
| HMC | 12 | 5e-5, 1e-4, 1.5e-4, 5e-4 | .025, .05, .1 | .1, .2, .3 | 12 / 8 |

Multivariate TPE uses six initial configurations, including the baseline and joint
regularization settings, then adapts using completed five-seed objectives. Repeated
parameter suggestions are not retrained. There is no single-seed screening or
cross-trial pruning based on a partial seed average. The initial budget is 68 distinct
configurations / 265 seed runs. Mental Arithmetic is fixed to the trial-0 parameters
and re-evaluated with validation checkpoint selection. Incomplete or failed runs never enter the objective.

Every trial starts from the same pretrained KNN37 checkpoint. All optimizer groups
use the common LR from the first update. Architecture, head sizes, batch sizes and
cosine horizon are retained (20 epochs TUAB, 50 elsewhere). Multiclass label smoothing
is fixed at .1, binary at 0; no mixup. Per-seed validation patience can stop a run only after
the dataset's minimum observation. Every metric is reported from the same selected
checkpoint, with population SD and the five individual seed values.

Each GPU job requests 2 CPUs, 20 GiB of host RAM and 4 hours; each dataset has one five-seed array, at most two
concurrent tasks. A timed-out or failed missing seed gets one bounded retry, resuming
the saved optimizer, scheduler and RNG state. A second failure requires attention.
Other datasets continue. GPU workers do not import Optuna or write its database.

The CPU controller serializes its calls with a file lock and owns the per-dataset
Optuna SQLite databases. Sampler state survives controller restarts. Submission
intents make ambiguous sbatch responses stop for reconciliation instead of silently
submitting a duplicate. A CPU preflight on the exact source snapshot must pass before
GPU submissions. Monitoring uses a local CPU process; no GPU is reserved for it.

Final comparison tables are published only after all five datasets have a winner
with exactly five validated result.json files. Provisional trial progress is separate.
All outputs and the global results index explicitly record validation checkpoint
selection and test-informed trial selection.

A campaign may contain `dataset_controls.json` with version 1 and per-dataset
`fix_existing_trial` entries. A fixed dataset validates its existing five seed
artifacts and reports `fixed`; it cannot submit a new trial or retry. Other datasets
continue normally. A campaign ending with fixed settings is recorded separately
without publishing an automatic final comparison aggregate. This control preserves
historical study results and does not turn test-informed scores into independent
evaluation results.
