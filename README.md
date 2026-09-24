# EEG foundation model — final mask-55 release

This branch contains the final model and training path only: geometry-tubelet
masking at 55%, a patch-dimension fusion gate, a three-stage S→T/T→S encoder,
and a depth-2 MAE decoder. The model/training logic matches the final experiment;
ablation models, campaign scripts, logs, checkpoints and data are not included.

## Setup and data

Use Python 3.9 with the dependencies in `requirements.txt` (CUDA-enabled
PyTorch 2.0.1 for GPU training). `preprocess_pretrain.py` converts licensed
TUH/TUEG EDF recordings to an LMDB with `__keys__` and 19-channel, 30-patch,
200-sample records. Downstream LMDBs must follow the split/record contracts in
`src/data/datasets/`. Obtain datasets under their own licenses.

Only `configs/pretrain.yaml` and `configs/downstream.yaml` are supplied.
`path` is a placeholder; pass actual locations through CLI flags. Outputs
default to the ignored `outputs/` directory.

```bash
python preprocess_pretrain.py --raw-root /path/to/tueg/edf --output /path/to/pretrain-lmdb
python pretrain.py --data-dir /path/to/pretrain-lmdb --output outputs/pretrain
python finetune.py --dataset chb --data-dir /path/to/chb-lmdb \
  --checkpoint /path/to/pretrain-checkpoint.pth --seed 42
```

For a distributed pretraining launch, use `torchrun` and add `--distributed`.
`--smoke` performs a bounded diagnostic pass; it is not a final result.
The downstream command accepts `--learning-rate`, `--weight-decay`,
`--dropout`, and `--output` overrides. For a binary dataset other than CHB,
pass its training-set counts with `--class-counts NEGATIVE POSITIVE`.
Select checkpoints and hyperparameters on validation balanced accuracy only;
test metrics are reporting-only. Run independent seeds by changing `--seed`.

Final-campaign checkpoints with the canonical MJDE/SHPE ablation wrapper are
supported: their `encoder.core.*` namespace is mapped to `encoder.*` and all
backbone tensors are then loaded strictly. Other ablation variants are not
supported in this branch.
