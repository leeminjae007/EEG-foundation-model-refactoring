# GR2 dimension gate, decoder 2, masking 60%

This preset changes only the mask ratio from the preceding
`gr2-d2-patch-dimension` profile. Geometry remains 35–75 degrees and 2–10 seconds;
the MJDE encoder remains three stages / twelve blocks, with patch-feature gates.
Decoder depth is two. Pretrain uses 40 epochs, seed 42, batch 128 per GPU,
four GPUs, independent rank RNG, AdamW LR 5e-4 → 1e-5 and weight decay 0.05.

## Launch on BigPurple

```bash
cd /gpfs/data/oermannlab/users/ml10266/workspace/EEG-foundation-model-refactoring
module load git/2.49.0
conda activate eeg-foundation-model-cu118
python scripts/launch_experiment_pretrain.py gr2-d2-patch-dimension-mask60
```

For a different experiment folder name, add
`--preset gr2-d2-patch-dimension-mask60`. An arbitrary name alone uses the
launcher's original geometry preset, so retain the explicit preset in that case.
`--prepare-only` snapshots the source and writes all configs without submitting.

One command submits:

1. Pretrain: A100 ×4 across 1–4 nodes, 4 CPUs and 32 GiB RAM per GPU,
   `a100_short,a100_long`, 72 hours, no automatic timeout continuation.
   The job exits successfully only after the final checkpoint passes strict
   config, epoch, model-weight, optimizer/scheduler and four-rank RNG checks.
2. Twelve downstream arrays, five seeds each (42, 696, 1001, 1234, 3407), with
   `afterok:<pretrain>` and `--kill-on-invalid-dep=yes`. There is no held-array
   release service or polling monitor. A failed pretrain prevents downstream.
3. One short CPU-only `afterany` finalizer after all downstream arrays finish.
   It runs once, records failures/missing artifacts and runtime measurements,
   and publishes only after every dataset has exactly five complete results
   and validation-selected weights. It does not launch or monitor training.

The source/config snapshots and job IDs are in the printed experiment folder's
`manifest.json`. `completion.json` and `downstream_runtime.json` retain completion
and timing evidence. Published tables, individual seeds and the global index
are under the repository's `outputs/results/`.

## Downstream resources and settings

The mask60 main experiment uses A100 ×1, CPU ×2, RAM 32 GiB per downstream seed,
as explicitly requested. The four earlier GR2 main profiles also retain A100.
Later ablations retain the separate L40S policy. CPU thread limits and GPU UUID
binding are applied in each worker. Batch sizes and accumulation match the
existing templates: most datasets use batch 64; ISRUC uses sequence batch 2
with accumulation 32; TUAB uses batch 64 with accumulation 8. Each starts at
its configured common base LR without downstream warmup.

| Dataset | Partitions | Time limit (estimate) |
|---|---|---|
| CHB-MIT | a100_short,a100_long | 12 h |
| SIENA | a100_dev,a100_short,a100_long | 4 h |
| PHYSIONET-MI | a100_dev,a100_short,a100_long | 4 h |
| TUEV | a100_short,a100_long | 12 h |
| TUAB | a100_short,a100_long | 12 h |
| FACED | a100_short,a100_long | 12 h |
| SEED-V | a100_dev,a100_short,a100_long | 4 h |
| Mental Arithmetic | a100_dev,a100_short,a100_long | 4 h |
| ISRUC | a100_short,a100_long | 12 h |
| HMC | a100_short,a100_long | 12 h |
| TUSL | a100_dev,a100_short,a100_long | 4 h |
| TUSZ | a100_dev,a100_short,a100_long | 4 h |

Dev has a four-hour limit; twelve-hour requests use short/long. These are
scheduling estimates, not empirical five-seed
averages. The full-run runtime report labels a measured mean only when all five
observations exist. TUSZ's required binary class counts are [28670, 12842],
verified against the processed training manifest; no validation/test counts
are used. Its smoothing is zero, as in other binary tasks.

## Real-data smoke

Use a fresh prepared experiment, then run:

```bash
python scripts/submit_experiment_smoke.py --experiment <prepared-experiment-folder>
python scripts/audit_experiment_smoke.py --experiment <prepared-experiment-folder>
```

The pretrain smoke runs for approximately 300 seconds in the training loop at
the configured full batch and four-rank Slurm topology. Startup, checkpoint
serialization and Slurm waiting are additional. The dependent L40S array checks
seed 42 for all twelve datasets at their actual batch/accumulation settings,
including four optimizer updates, bounded validation/test evaluation, selected
checkpoint saving and final gate export. Other seeds' configs are also checked.

Smoke outputs are explicitly partial and cannot resume production pretraining.
Bounded downstream evaluation writes `smoke_result.json`, never `result.json`.
Smoke metrics are implementation checks, not scientific performance results.
For an available diagnostic L40S allocation, add `--gpu l40s` to the smoke
submission. Production pretraining remains A100; an alternate GPU type is
rejected by the production controller outside explicitly marked smoke runs.

## Validation completed on 2026-09-19

- Pretrain smoke job `27602508`: L40S ×4 on three nodes, 438 optimizer updates,
  302.84 seconds in training (Slurm elapsed 5:17), target/context 342/228.
  All four CUDA probes passed; peak reserved GPU memory was 21.60 GiB.
- Dependent downstream smoke array `27602509`: all twelve seed-42 tasks
  completed successfully, with configured batches/accumulation and four updates
  each, bounded validation/test, checkpoint and gate-output checks.
- All smoke weights remained finite, the three gate projections trained,
  strict loading succeeded, and partial pretrain resume was rejected.
- The final-checkpoint verifier also accepted an existing complete 40-epoch,
  86,680-step checkpoint with four independent rank RNG states.
- 32 targeted launcher, publication, gate and RNG tests passed locally.
- Smoke has no production performance claims. Production A100 allocation and
  full-epoch completion remain to be observed after the user runs the command.
