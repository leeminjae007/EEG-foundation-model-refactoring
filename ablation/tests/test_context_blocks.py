from copy import deepcopy

import pytest
import torch

from ablation.config import load_config, resolve_ablation
from ablation.models import build_backbone
from ablation.encoders.context_blocks import (
    labram_block, cbramod_block, csbrain_block, masked_mha, window_attention,
)


def backbone(name, depth=1):
    config = resolve_ablation(load_config("ablation/configs/encoder_" + name + ".yaml"))
    if name != "mjde":
        config["ablation"]["depth"] = depth
    return build_backbone(config).eval()


@pytest.mark.parametrize("name,count", [("mjde", 5787200), ("labram", 5794960),
                                        ("cbramod", 4831200), ("csbrain", 8208000)])
def test_twelve_blocks_and_only_active_parameters(name, count):
    model = backbone(name, 12)
    assert sum(p.numel() for p in model.encoder.parameters()) == count
    if name == "mjde":
        core = model.encoder.core
        assert sum(len(getattr(core, n)) for n in ("s2t_spatial", "s2t_temporal", "t2s_spatial", "t2s_temporal")) == 12
    else:
        core = model.encoder.core.core
        assert len(core.blocks if name == "labram" else core.encoder.layers) == 12


@pytest.mark.parametrize("name", ["labram", "cbramod", "csbrain"])
def test_native_block_output_and_gradients_without_mask(name):
    model = backbone(name)
    core = model.encoder.core.core
    original = core.blocks[0] if name == "labram" else core.encoder.layers[0]
    adapted = deepcopy(original)
    x = torch.randn(2, 19, 10, 200)
    if name == "labram":
        x = x.flatten(1, 2)
    x = x.requires_grad_()
    y = x.detach().clone().requires_grad_()
    mask = torch.ones(x.shape[:-1], dtype=torch.bool)
    expected = original(x)
    actual = {"labram": labram_block, "cbramod": cbramod_block, "csbrain": csbrain_block}[name](adapted, y, mask)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(y.grad, x.grad, rtol=2e-4, atol=2e-7)
    params = dict(adapted.named_parameters())
    for key, p in original.named_parameters():
        assert p.grad is not None and params[key].grad is not None, key
        torch.testing.assert_close(params[key].grad, p.grad, rtol=2e-4, atol=2e-7)


@pytest.mark.parametrize("name", ["mjde", "labram", "cbramod", "csbrain"])
@pytest.mark.parametrize("patches", [1, 6])
def test_masked_values_and_missing_rows_cannot_affect_context(name, patches):
    model = backbone(name)
    tokens = torch.randn(2, 19, patches, 200, requires_grad=True)
    visible = torch.rand(2, 19, patches) > .5
    visible[0] = False
    visible[:, 0] = False
    changed = tokens.detach().clone()
    changed[~visible] = torch.randn_like(changed[~visible]) * 1000
    expected = model.encoder(tokens, visible)
    actual = model.encoder(changed, visible)
    assert torch.isfinite(expected).all()
    assert torch.count_nonzero(expected[~visible]) == 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().sum().backward()
    assert torch.isfinite(tokens.grad).all()
    assert torch.count_nonzero(tokens.grad[~visible]) == 0


def test_masked_attention_equals_physically_removed_keys():
    module = torch.nn.MultiheadAttention(16, 4, batch_first=True).eval()
    tokens = torch.randn(2, 7, 16)
    visible = torch.tensor([[True, False, True, False, False, True, False], [False] * 7])
    result = masked_mha(module, tokens, visible)
    packed = tokens[0, visible[0]][None]
    expected = module(packed, packed, packed, need_weights=False)[0]
    torch.testing.assert_close(result[0, visible[0]], expected[0], rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(result[1]) == 0


def test_window_padding_does_not_act_as_a_key():
    block = backbone("csbrain").encoder.core.core.encoder.layers[0]
    tokens = torch.randn(1, 19, 6, 200)
    visible = torch.ones(1, 19, 6, dtype=torch.bool)
    actual = window_attention(block, tokens, visible)
    # With six patches, offsets 1..4 have only one real window; padding
    # must not dilute that one key's attention probability.
    for offset in range(1, 5):
        packed = tokens[:, :, offset, :].reshape(19, 1, 200)
        expected = block.inter_window_attn(packed, packed, packed, need_weights=False)[0]
        torch.testing.assert_close(actual[0, :, offset], expected[:, 0], rtol=1e-5, atol=1e-6)


def test_legacy_dense_checkpoint_rejected_at_construction():
    config = resolve_ablation(load_config("ablation/configs/encoder_mjde.yaml"))
    config["ablation"]["mask_mode"] = "dense_zero"
    with pytest.raises(ValueError, match="dense_zero"):
        build_backbone(config)
