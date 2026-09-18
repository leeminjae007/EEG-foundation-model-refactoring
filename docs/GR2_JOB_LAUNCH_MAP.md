# GR2 pretrain / downstream launch map

This map separates Git-tracked launch code from the generated `worker.sh`
files used by already submitted jobs.  Generated workers live in a campaign
snapshot and are intentionally not reused by another account: they contain
the submitting user's absolute output, environment, and Slurm-account paths.

## Submitted pretraining jobs

| Job | Arm | GPU | Git-tracked config | Git-tracked submit controller | Generated worker currently used on BigPurple |
| --- | --- | --- | --- | --- | --- |
| `27525946` | Geometry, decoder depth 4 | A100 ×4 | `configs/pretrain_gr2_geometry.yaml` | `scripts/gr2_pretrain_campaign.py` | `outputs/mjde_gr2_geometry_20260917_171816/worker.sh` |
| `27534936` | Static, decoder depth 2 | L40S ×4 | generated from `configs/pretrain_gr2_geometry.yaml` | `scripts/gr2_decoder2_l40s_campaign.py` | `outputs/gr2_decoder2_l40s_gates_20260918_0015/worker.sh` |
| `27534938` | Patch scalar, decoder depth 2 | L40S ×4 | `configs/pretrain_gr2_patch_scalar.yaml` | `scripts/gr2_decoder2_l40s_campaign.py` | `outputs/gr2_decoder2_l40s_gates_20260918_0015/worker.sh` |
| `27534940` | Patch dimension, decoder depth 2 | L40S ×4 | `configs/pretrain_gr2_patch_feature.yaml` | `scripts/gr2_decoder2_l40s_campaign.py` | `outputs/gr2_decoder2_l40s_gates_20260918_0015/worker.sh` |

For a new L40S decoder-2 campaign, use the Git controller to create a fresh
snapshot and only then submit it:

```bash
python scripts/gr2_decoder2_l40s_campaign.py prepare \
  --folder outputs/<your-campaign-name>
python scripts/gr2_decoder2_l40s_campaign.py submit \
  --folder outputs/<your-campaign-name>
```

The controller submits all three L40S arms, each as four distributed ranks,
and attaches CPU-only timeout/preemption callbacks.  It does not submit
downstream jobs.

## Downstream submission

No downstream Slurm job has been submitted for these four arms.  This is
intentional: a downstream campaign may be prepared only after the matching
epoch-40 pretraining checkpoint passes strict loading, config/hash, optimizer,
scheduler, and four-rank RNG audits.  Do not submit from a pending or partial
checkpoint.

The existing historical downstream controller is
`scripts/submit_ablation_downstream_all.py`.  It is tied to old ablation
checkpoint paths and must **not** be used for these GR2 arms.  A GR2-specific
downstream campaign must set the verified checkpoint path in a fresh set of
per-dataset, five-seed configs and write results below `outputs/results/` in
the required dataset order.  It must select checkpoints by validation, never
by test results.
