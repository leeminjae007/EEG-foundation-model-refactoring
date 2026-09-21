# Decoder-depth-4 patch-dimension A100 runs

Both production profiles use `mae.decoder_depth: 4` and
`encoder.fusion_gate: patch_feature`. Pretraining uses four A100s; every
downstream seed uses one A100. The two profiles differ only in
`masking.mask_ratio` (`0.55` versus `0.60`).

Run from the shared checkout after activating the project environment. Set a
results root writable by the Unix account that submits the jobs.

```bash
cd /gpfs/data/oermannlab/users/ml10266/workspace/EEG-foundation-model-refactoring
module load git/2.49.0
conda activate eeg-foundation-model-cu118
export EEGFM_RESULTS_ROOT=/gpfs/data/oermannlab/users/$USER/workspace/eegfm/results
mkdir -p "$EEGFM_RESULTS_ROOT"
```

## Short A100 smoke for the new mask-55 profile

Use a disposable, prepare-only experiment. This submits a five-minute,
four-A100 pretrain smoke followed by one four-batch A100 check for each of the
twelve downstream datasets.

```bash
SMOKE55=$(python scripts/launch_experiment_pretrain.py \
  gr2-d4-patch-dimension-mask55-smoke \
  --preset gr2-d4-patch-dimension-mask55 \
  --results-root "$EEGFM_RESULTS_ROOT" \
  --prepare-only)
python scripts/submit_experiment_smoke.py --experiment "$SMOKE55" --gpu a100
squeue -u "$USER"
```

After both smoke jobs finish:

```bash
python scripts/audit_experiment_smoke.py --experiment "$SMOKE55"
```

## Production mask-55 pretrain and downstream

This submits pretraining, twelve dependent five-seed downstream arrays, and a
CPU-only finalizer. Downstream starts only after pretraining succeeds.

```bash
python scripts/launch_experiment_pretrain.py \
  gr2-d4-patch-dimension-mask55 \
  --results-root "$EEGFM_RESULTS_ROOT"
```

## Rerun mask-60 downstream on A100 without repeating pretraining

Point `OLD60` at the completed mask-60 experiment. The new experiment gets
fresh downstream output directories, so the earlier L40S results are retained.

```bash
OLD60=/gpfs/data/oermannlab/users/ml10266/workspace/eegfm/results/260920-2314-gr2-d4-patch-dimension-mask60
CKPT60="$OLD60/pretrain/checkpoint-epoch-0040.pth"
test -r "$CKPT60"

python scripts/smoke_existing_downstream.py \
  --experiment "$OLD60" \
  --checkpoint "$CKPT60" \
  --dataset tusz \
  --seed 42 \
  --batches 4 \
  --smoke-root "$EEGFM_RESULTS_ROOT/mask60-a100-downstream-smoke"

EXP60=$(python scripts/launch_experiment_pretrain.py \
  gr2-d4-patch-dimension-mask60-a100-rerun \
  --preset gr2-d4-patch-dimension-mask60 \
  --results-root "$EEGFM_RESULTS_ROOT" \
  --prepare-only)
python scripts/submit_experiment_downstream.py \
  --experiment "$EXP60" \
  --checkpoint "$CKPT60"
python scripts/finalize_experiment.py --experiment "$EXP60" --submit
```

The full mask-60 rerun requests A100 for all 60 downstream runs. The smoke and
production commands intentionally use separate output folders.
