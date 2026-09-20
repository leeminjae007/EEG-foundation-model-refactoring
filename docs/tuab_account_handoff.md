# TUAB account handoff

Use the PyTorch 2.0.1 / CUDA 11.8 environment. Run from the pulled repository on
BigPurple under the account that should own the remaining TUAB jobs:

```bash
conda activate eeg-foundation-model-cu118
python scripts/launch_tuab_handoff.py launch \
  /gpfs/data/oermannlab/users/ml10266/workspace/EEG-foundation-model-refactoring/outputs/experiments/tuab-repair-20260919-2350
```

The launcher checks each original job's live Slurm state, retains running and
completed seeds, and resumes only terminal, unfinished seeds. Pending originals
must first be cancelled by their owner. Outputs and a pinned Git source snapshot
are created under `/gpfs/data/oermannlab/users/<current-user>/workspace/eegfm/outputs/experiments/`.
Use `--output-root /your/writable/root` to change that base directory, or
`--prepare-only` to validate and copy without submitting any jobs.

Training settings remain unchanged; only the output path is relocated. The
complete optimizer, scheduler and RNG states are restored. Existing validation
weights are copied and checked by SHA256. Each unfinished seed uses one A100,
two CPUs, 32 GiB RAM and a four-hour limit on `a100_dev,a100_short,a100_long`.
All remaining seeds can run concurrently, subject to Slurm allocation limits.

Submission prints an exact `attach-tusz` command. Run that command as `ml10266`,
the owner of the existing TUSZ arrays. It points all four arrays at the new
CPU completion audit and then cancels the old held TUAB controller. Slurm does
not allow another account to update these existing jobs.

The audit waits for retained running jobs and all newly submitted arrays. It
publishes only after all four campaigns have exactly five complete seeds,
readable results and matching validation-selected weights. TUSZ starts only
after that audit succeeds, with five concurrent tasks per array. Failures or
missing artifacts block publication and TUSZ. The handoff launcher does not
automatically resubmit failed or timed-out new jobs; inspect `completion.json`
and logs before retrying. A per-account claim under `outputs/handoffs/` prevents
repeat submission of the same source campaign to the same output root. Do not
launch the same handoff from multiple accounts or output roots.
