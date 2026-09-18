"""Count encoder parameters and dominant inference FLOPs on a fixed EEG input.

FLOPs use two operations per multiply-accumulate. Linear/convolution modules and
multi-head-attention projections/products are counted; FFT, normalization,
activation, softmax, and elementwise bookkeeping are excluded.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import yaml
from torch import nn

ROOT = Path(os.environ.get("EEG_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))


class Counter:
    def __init__(self, model: nn.Module):
        self.macs = 0
        self.by_kind: dict[str, int] = {}
        self.handles = []
        for module in model.modules():
            if isinstance(module, nn.MultiheadAttention):
                self.handles.append(module.register_forward_hook(self._mha))
            elif isinstance(module, nn.Linear):
                # MHA's out_proj is invoked functionally, so the MHA hook counts it.
                if not any(module is mha.out_proj for mha in model.modules()
                           if isinstance(mha, nn.MultiheadAttention)):
                    self.handles.append(module.register_forward_hook(self._linear))
            elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                self.handles.append(module.register_forward_hook(self._conv))

    def _add(self, kind: str, value: int):
        self.macs += value
        self.by_kind[kind] = self.by_kind.get(kind, 0) + value

    def _linear(self, module, inputs, output):
        x = inputs[0]
        vectors = x.numel() // module.in_features
        self._add("linear", vectors * module.in_features * module.out_features)

    def _conv(self, module, inputs, output):
        kernel = 1
        for size in module.kernel_size:
            kernel *= size
        self._add("conv", output.numel() * module.in_channels // module.groups * kernel)

    def _mha(self, module, inputs, output):
        query = inputs[0]
        batch, tokens, dim = query.shape
        projections = 4 * batch * tokens * dim * dim
        attention = 2 * batch * tokens * tokens * dim
        self._add("mha_projection", projections)
        self._add("mha_attention", attention)

    def close(self):
        for handle in self.handles:
            handle.remove()


def profile(model: nn.Module, arguments, attention_macs: int = 0):
    model.eval()
    counter = Counter(model)
    with torch.no_grad():
        model(*arguments)
    counter.close()
    return {
        "params": sum(p.numel() for p in model.parameters()),
        "macs": counter.macs + attention_macs,
        "flops": 2 * (counter.macs + attention_macs),
        "macs_by_kind": counter.by_kind,
        "extra_attention_macs": attention_macs,
    }


def reve_flops(channels: int, samples: int, width: int, depth: int, heads: int):
    """Official REVE GEGLU encoder arithmetic with 200-sample/20-overlap patches."""
    tokens = channels * ((samples - 200) // 180 + 1)
    inner = heads * 64
    hidden = int(width * 2.66)
    embedding_macs = tokens * (200 * width + 4 * width)
    projection_macs = depth * tokens * (4 * width * inner + 3 * width * hidden)
    attention_macs = depth * 2 * tokens * tokens * inner
    return {
        "tokens": tokens,
        "width": width,
        "depth": depth,
        "macs": embedding_macs + projection_macs + attention_macs,
        "flops": 2 * (embedding_macs + projection_macs + attention_macs),
    }


def main():
    torch.set_num_threads(1)
    with (ROOT / "configs/pretrain.yaml").open() as handle:
        config = yaml.safe_load(handle)
    names = config["data"]["channel_names"]
    channels, patches, samples = len(names), 30, 200

    from src.model import EEGEncoder, PretrainModel

    ours = EEGEncoder(config)
    signal = torch.zeros(1, channels, patches * samples)
    positions = ours.default_channel_coordinates.unsqueeze(0)
    valid_channel = torch.ones(1, channels, dtype=torch.bool)
    visible = torch.ones(1, channels, patches, dtype=torch.bool)
    dim = config["encoder"]["embed_dim"]
    attention_macs = 6 * 2 * patches * channels * channels * dim
    attention_macs += 6 * 2 * channels * patches * patches * dim
    results = {"mjbrain": profile(ours, (signal, positions, valid_channel, visible), attention_macs)}
    results["mjbrain"]["pretrain_params_with_decoder"] = sum(
        parameter.numel() for parameter in PretrainModel(config, torch.device("cpu")).parameters()
    )

    sys.path.insert(0, str(ROOT / "ablation/vendor/csbrain"))
    from ablation.montage import region_order
    from models.CSBrain import CSBrain

    regions, order = region_order(names)
    csbrain = CSBrain(n_layer=12, brain_regions=regions, sorted_indices=order)
    results["csbrain"] = profile(csbrain, (torch.zeros(1, channels, patches, samples),))
    for label, width, depth, heads in (
        ("reve_small", 512, 4, 8),
        ("reve_base", 512, 22, 8),
        ("reve_large", 1216, 22, 19),
    ):
        results[label] = reve_flops(channels, patches * samples, width, depth, heads)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
