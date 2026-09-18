"""Dynamic-route semantics, fair initialization, masking and training continuity."""
from copy import deepcopy

import pytest
import torch
import yaml

from src.encoder import Encoder
from src.model import PretrainModel
from src.modules.fusion import PatchFusionGate
from src.modules.masking import make_masks
from src.training.checkpoint import load_checkpoint, rng_state, save_checkpoint
from src.training.diagnostics import capture, measurements
from src.training.runtime import ROOT

torch.set_num_threads(2)


def config(mode="static_feature"):
    path = "pretrain_gr2_geometry" if mode == "static_feature" else "pretrain_gr2_" + mode
    return yaml.safe_load((ROOT / ("configs/" + path + ".yaml")).read_text(encoding="utf-8"))


@pytest.mark.parametrize("mode,width,count", [("patch_scalar", 1, 1203), ("patch_feature", 200, 240600)])
def test_only_gate_changes_and_common_initialization_is_exact(mode, width, count):
    baseline = config()
    changed = config(mode)
    restored = deepcopy(changed)
    restored["encoder"].pop("fusion_gate")
    restored["runtime"]["output"] = baseline["runtime"]["output"]
    assert restored == baseline
    torch.manual_seed(42)
    original = PretrainModel(baseline, torch.device("cpu"))
    original_rng = torch.get_rng_state()
    torch.manual_seed(42)
    dynamic = PretrainModel(changed, torch.device("cpu"))
    assert torch.equal(torch.get_rng_state(), original_rng)
    original_state = original.state_dict()
    for name, value in dynamic.state_dict().items():
        if ".patch_gates." in name:
            assert torch.count_nonzero(value) == 0
        else:
            assert torch.equal(value, original_state[name]), name
    encoder = dynamic.backbone.encoder
    assert sum(p.numel() for p in encoder.patch_gates.parameters()) == count
    assert len(encoder.patch_gates) == 3
    assert all(g.weight.shape == (width, 400) for g in encoder.patch_gates)
    assert all(len(getattr(encoder, name)) == 3 for name in
               ("s2t_spatial", "s2t_temporal", "t2s_temporal", "t2s_spatial"))
    signals = torch.randn(1, 19, 800)
    masks = make_masks(1, 19, 4, changed["masking"], torch.device("cpu"), dynamic.mask_coordinates)
    original.eval()
    dynamic.eval()
    expected = original(signals, masks)[0]
    actual = dynamic(signals, masks)[0]
    assert torch.equal(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    reference_parameters = dict(original.named_parameters())
    for name, parameter in dynamic.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        if ".patch_gates." in name:
            assert parameter.grad.abs().sum() > 0, name
        else:
            assert torch.equal(parameter.grad, reference_parameters[name].grad), name


@pytest.mark.parametrize("width", [1, 4])
def test_gate_is_input_conditioned_and_shared_across_positions(width):
    gate = PatchFusionGate(4, width)
    a, b = torch.randn(2, 3, 5, 4), torch.randn(2, 3, 5, 4)
    visible = torch.ones(2, 3, 5, dtype=torch.bool)
    assert torch.equal(gate(a, b, visible), torch.full((2, 3, 5, width), .5))
    with torch.no_grad():
        gate.weight[:, 0] = torch.arange(1, width + 1) * .1
        gate.weight[:, 4] = -.15
    before = gate(a, b, visible)
    modified = a.clone()
    modified[0, 1, 2, 0] += 1
    after = gate(modified, b, visible)
    assert (after[0, 1, 2] > before[0, 1, 2]).all()
    unchanged = torch.ones_like(visible)
    unchanged[0, 1, 2] = False
    assert torch.equal(before[unchanged], after[unchanged])
    assert torch.equal(gate(a.flip(2), b.flip(2), visible), before.flip(2))
    assert before.shape[-1] == width
    if width > 1:
        assert not torch.equal(before[..., 0], before[..., 1])


@pytest.mark.parametrize("mode", ["patch_scalar", "patch_feature"])
def test_masked_channels_empty_rows_and_hidden_values_cannot_leak(mode):
    cfg = dict(config(mode)["encoder"], embed_dim=8, spatial_heads=2, temporal_heads=2,
               dropout=0., attention_dropout=0.)
    encoder = Encoder(cfg).eval()
    for gate in encoder.patch_gates:
        torch.nn.init.normal_(gate.weight, std=.1)
    tokens = torch.randn(2, 3, 4, 8, requires_grad=True)
    visible = torch.ones(2, 3, 4, dtype=torch.bool)
    visible[0, 1] = False
    visible[0, :, 2] = False
    visible[1] = False
    capture(encoder, True)
    output = encoder(tokens, visible)
    diagnostics = measurements(encoder)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[~visible]) == 0
    changed = tokens.detach().clone()
    changed[~visible] = torch.randn_like(changed[~visible]) * 100000
    assert torch.equal(output, encoder(changed, visible))
    output.square().sum().backward()
    assert torch.count_nonzero(tokens.grad[~visible]) == 0
    assert tokens.grad[visible].abs().sum() > 0
    counts = [value for key, value in diagnostics.items() if key.endswith("gate/visible_patches")]
    assert counts == [int(visible.sum())] * 3
    assert all(torch.isfinite(p.grad).all() for p in encoder.patch_gates.parameters())


@pytest.mark.parametrize("mode", ["patch_scalar", "patch_feature"])
def test_optimizer_resume_restores_gate_and_rng_exactly(tmp_path, mode):
    cfg = config(mode)
    settings = dict(cfg["encoder"], embed_dim=8, spatial_heads=2, temporal_heads=2)
    model = Encoder(settings)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10)
    tokens, visible = torch.randn(2, 3, 4, 8), torch.ones(2, 3, 4, dtype=torch.bool)
    def step(net, optimizer, scheduler):
        optimizer.zero_grad(set_to_none=True)
        loss = (net(tokens, visible) - .2).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return loss.detach()
    step(model, opt, sched)
    path = tmp_path / "last.pth"
    save_checkpoint(path, model, opt, sched, 1, cfg, [rng_state(torch.device("cpu"))], {"step": 1})
    expected_loss = step(model, opt, sched)
    expected = deepcopy(model.state_dict())
    restored = Encoder(settings)
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=5e-4)
    restored_sched = torch.optim.lr_scheduler.CosineAnnealingLR(restored_opt, 10)
    load_checkpoint(path, restored, restored_opt, restored_sched, torch.device("cpu"), 0,
                    expected_masking=cfg["masking"], expected_pretrain_rng="independent")
    assert torch.equal(step(restored, restored_opt, restored_sched), expected_loss)
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in expected.items())
    other = dict(settings, fusion_gate="patch_feature" if mode == "patch_scalar" else "patch_scalar")
    with pytest.raises(RuntimeError):
        Encoder(other).load_state_dict(expected, strict=True)


@pytest.mark.parametrize("mode", ["patch_scalar", "patch_feature"])
def test_native_pretrain_weight_loads_into_downstream_and_trains_gates(tmp_path, mode):
    from src.training.engine import build_finetune, optimizer_groups
    from src.data.datasets.registry import get_dataset_spec
    from src.data.electrode_geometry import resolve_channel_coordinates
    cfg = config(mode)
    pretrained = PretrainModel(cfg, torch.device("cpu"))
    path = tmp_path / "pretrain.pth"
    torch.save(dict(model=pretrained.state_dict(), config=cfg), path)
    downstream = yaml.safe_load((ROOT / "configs/downstream/gr9-1_tuev_seed42.yaml").read_text(encoding="utf-8"))
    downstream["model"].update(checkpoint=str(path), head_hidden_tokens=1)
    model = build_finetune(downstream)
    spec = get_dataset_spec(downstream["data"]["dataset"])
    coordinates, valid = resolve_channel_coordinates(spec.dataset_class.channel_names)
    output = model(torch.randn(1, spec.num_channels, spec.signal_length), coordinates[None], valid[None])
    assert output.shape == (1, spec.num_outputs)
    torch.nn.functional.cross_entropy(output, torch.zeros(1, dtype=torch.long)).backward()
    parameters = [p for group in optimizer_groups(model, downstream["optimization"]) for p in group["params"]]
    assert len(parameters) == len({id(p) for p in parameters})
    assert {id(p) for p in parameters} == {id(p) for p in model.parameters()}
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.encoder.patch_gates.parameters())


def test_bad_modes_and_adapters_cannot_silently_discard_dynamic_gate():
    from ablation.config import resolve_ablation
    from ablation.models import build_pretrain
    with pytest.raises(ValueError, match="fusion_gate"):
        Encoder(dict(config()["encoder"], fusion_gate="patch_typo"))
    cfg = resolve_ablation(config("patch_scalar"))
    cfg["ablation"]["encoder"] = "mjde_lite"
    with pytest.raises(ValueError, match="full MJDE"):
        build_pretrain(cfg, torch.device("cpu"))


@pytest.mark.parametrize("mode", ["patch_scalar", "patch_feature"])
def test_bfloat16_autocast_keeps_gate_and_branch_gradients_finite(mode):
    cfg = dict(config(mode)["encoder"], embed_dim=8, spatial_heads=2, temporal_heads=2,
               dropout=0., attention_dropout=0.)
    encoder = Encoder(cfg)
    tokens = torch.randn(2, 3, 4, 8, requires_grad=True)
    visible = torch.ones(2, 3, 4, dtype=torch.bool)
    visible[0, 1] = False
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = encoder(tokens, visible)
        loss = (output.float() - torch.randn_like(output)).square().mean()
    loss.backward()
    assert torch.isfinite(output).all() and torch.count_nonzero(output[~visible]) == 0
    assert torch.isfinite(tokens.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               and p.grad.abs().sum() > 0 for p in encoder.patch_gates.parameters())
