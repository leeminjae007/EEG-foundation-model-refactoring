# BCIC-IV-2a beam search

User decision (2026-09-15): **warmup stays 0 for every candidate and final run**.

The current nearest3–7 epoch40 checkpoint is used throughout. This replaces the
unsubmitted single-recipe proposal. Existing submitted campaigns are separate.

## Search

| Parameter | Values |
| --- | --- |
| Tokenizer + encoder LR | 5e-5, 1e-4, 2e-4 |
| Head LR | 5e-5, 1e-4, 2e-4 |
| Weight decay | .005, .01, .05 |
| Head dropout | .1, .2, .3 |

Fixed: 50 epochs, warmup **0**, batch 64×1, minimum LR 1e-6, label smoothing .1,
original all-patch H4 head, full finetuning and all-epoch validation BAcc selection.

Start at LR 1e-4 for all groups, WD .01, dropout .2, plus its eight one-coordinate
neighbors. Each candidate uses seeds **42, 1234, 696**. Rank by mean best validation
BAcc, then population SD, then candidate ID. Keep the top two candidates seen so
far and expand their adjacent parameter values, excluding visited configurations.
The search ends after at most **three rounds / 25 distinct configurations**.
If the budget truncates an expansion, dimensions follow the table order and beam
parents alternate within each dimension. This is a bounded local search; it does
not establish a global optimum of the 81-point grid.

Candidates missing any of their three complete 50-epoch runs are ineligible.
All failures are preserved in each stage's `scores.json`.

The search does not construct or evaluate the test dataset. After selection,
`winner.json` is written before scheduling five final runs with seeds
**42, 1234, 696, 1001, 3407** and test evaluation enabled. The first three seeds
overlap the search seeds; the five-seed report is confirmation, not an independent
validation estimate. Test metrics never enter the search ranking.

## Execution

```bash
cd /gpfs/data/oermannlab/users/ml10266/workspace/EEG-founation-model
.venv/bin/python -m ablation.bciciv2a.campaign all
```

`all` snapshots the completed baseline source/configs, validates the exact
checkpoint, runs a real CPU train/validation smoke and submits round 0 plus its
dependent CPU controller. Controllers submit subsequent rounds and the final
five-seed campaign automatically. Repeating submission reuses its recorded job IDs.
An ambiguous Slurm response requires reconciliation before retrying.

GPU jobs use one L40S, 8 CPUs, 32 GB and four hours each, across
`gl40s_dev,gl40s_short,gl40s_long`, at most ten concurrent runs. Controllers use
two CPUs and 4 GB in `cpu_short,cpu_long`. Maximum budget: 75 search runs + five
final runs; the first round contains 27 runs.

The isolated snapshot engine has two small changes, both gating test access.
The external SEED-VIG file cleanup also requires removing its two obsolete
registry lines in this new snapshot; excluded missing files are recorded explicitly.
Training, loss, validation and checkpoint selection are copied from the completed
baseline. The repository's `src/` and other queued snapshots are untouched.

Artifacts: `outputs/bciciv2a_beam_w0_20260915/`:

- `manifest.json`: checkpoint/source hashes, fixed parameters, search budget.
- `stages/round_*/plan.json`: immutable candidate configs and seeds.
- `ranking.json`: current validation ranking.
- `winner.json`: selected configuration, written before final test runs.
- `final_summary.json`: final five-seed validation/test mean and population SD.
- `status.json`: latest controller stage/status.

## Submitted

2026-09-15 19:25 UTC: round 0 GPU array **27484467** (27 runs), dependent CPU
controller **27484468**. At 19:25:46 UTC the GPU runs were pending
`ReqNodeNotAvail, May be reserved for other job`; the controller was pending
`Dependency`. Eight local tests and the real CPU train/validation/checkpoint smoke
passed. All 27 submitted configs have warmup 0 and test evaluation disabled.

```bash
squeue -r -j 27484467,27484468
cat outputs/bciciv2a_beam_w0_20260915/status.json
```
