from copy import deepcopy

import pytest
import torch

from ablation.bootstrap import ROOT
from ablation.config import load_config, resolve_ablation
from ablation.models import build_pretrain, build_backbone
from ablation.montage import region_order
from ablation.sources import verify_sources, upstream, reve_position_source
from src.model import PretrainModel, FinetuneModel, SleepModel


def config(name):
    return resolve_ablation(load_config("ablation/configs/" + name + ".yaml"))


def test_upstream_bytes():
    # The removed SEED-VIG adapter is intentionally absent from the vendor set.
    assert verify_sources()["verified_files"] >= 26


def test_shpe_baseline_exact_output_and_gradients():
    settings = config("pe_shpe")
    torch.manual_seed(41)
    original = PretrainModel(settings, torch.device("cpu"))
    torch.manual_seed(41)
    adapted = build_pretrain(settings, torch.device("cpu"))
    original.eval()
    adapted.eval()
    signals = torch.randn(2, 19, 1200)
    expected, masks = original(signals)
    actual, _ = adapted(signals, masks)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    parameters = dict(adapted.named_parameters())
    for name, value in original.named_parameters():
        adapted_name = name.replace("backbone.encoder.", "backbone.encoder.core.")
        torch.testing.assert_close(parameters[adapted_name].grad, value.grad, rtol=0, atol=0)


def test_common_components_initialize_identically():
    torch.manual_seed(19)
    first = build_pretrain(config("encoder_mjde"), torch.device("cpu"))
    torch.manual_seed(19)
    second = build_pretrain(config("encoder_cbramod"), torch.device("cpu"))
    for component in ("tokenizer", "position"):
        expected = getattr(first.backbone, component).state_dict()
        for name, value in getattr(second.backbone, component).state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    for name, value in second.decoder.state_dict().items():
        torch.testing.assert_close(value, first.decoder.state_dict()[name], rtol=0, atol=0)


def test_cbramod_encoder_matches_original_stack():
    model = build_backbone(config("encoder_cbramod")).eval()
    core = model.encoder.core.core
    tokens = torch.randn(2, 19, 6, 200)
    mask = torch.ones(2, 19, 6, dtype=torch.bool)
    zeroed = tokens.masked_fill(~mask[..., None], 0)
    expected = core.encoder(zeroed).masked_fill(~mask[..., None], 0)
    torch.testing.assert_close(model.encoder(tokens, mask), expected, rtol=1e-5, atol=2e-6)


def test_acpe_calls_original_conv_and_excludes_target_content():
    settings = config("pe_acpe")
    model = build_backbone(settings).eval()
    position = model.position
    tokens = torch.randn(2, 19, 6, 200)
    mask = torch.rand(2, 19, 6) > .5
    coordinates = model.default_channel_coordinates
    expected = position.positional_encoding(tokens.masked_fill(~mask[..., None], 0).permute(0, 3, 1, 2))
    actual = position(tokens, coordinates, mask)
    torch.testing.assert_close(actual, expected.permute(0, 2, 3, 1), rtol=0, atol=0)
    assert position.positional_encoding[0].kernel_size == (19, 7)
    assert position.positional_encoding[0].groups == 200


def test_reve_original_formula_and_input_not_modified():
    model = build_backbone(config("pe_reve4d")).eval()
    position = model.position
    coordinates = model.default_channel_coordinates[None].expand(2, -1, -1).clone()
    before = coordinates.clone()
    module = reve_position_source()
    pos4d = module.FourierEmb4D.add_time_patch(coordinates * position.coordinate_scale, 6)
    expected = position.ln(position.fourier4d(pos4d) + position.mlp4d(pos4d)).reshape(2, 19, 6, 200)
    torch.testing.assert_close(position(coordinates, 2, 6, torch.float32), expected, rtol=0, atol=0)
    torch.testing.assert_close(coordinates, before, rtol=0, atol=0)


def test_channel_ids_follow_names_not_column_index():
    settings = config("pe_channel_id")
    model = build_backbone(settings).eval()
    position = model.position
    names = settings["data"]["channel_names"]
    coordinates = model.default_channel_coordinates
    expected = position(coordinates, 2, 6, torch.float32)
    order = torch.randperm(len(names))
    position.set_channels([names[index] for index in order])
    actual = position(coordinates[order], 2, 6, torch.float32)
    torch.testing.assert_close(actual, expected[:, order], rtol=0, atol=0)
    position.set_channels(["T7", "T8", "P7", "P8"])
    canonical_indices = position.indices.clone()
    position.set_channels(["T3", "T4", "T5", "T6"])
    assert torch.equal(canonical_indices, position.indices)


def test_mjde_lite_removes_two_stages():
    full = build_pretrain(config("encoder_mjde"), torch.device("cpu"))
    lite = build_pretrain(config("encoder_mjde_lite"), torch.device("cpu"))
    core = lite.backbone.encoder.core
    assert core.fusion_gates.shape == (1, 200)
    assert len(core.s2t_spatial) == len(core.t2s_temporal) == 1
    assert sum(p.numel() for p in core.parameters()) < sum(p.numel() for p in full.backbone.encoder.parameters()) / 2


@pytest.mark.parametrize("name,order", [
    ("encoder_mjde_s2t6", "s2t"),
    ("encoder_mjde_t2s6", "t2s"),
])
def test_single_path_mjde_uses_six_stages_and_all_twelve_blocks(name, order):
    baseline = build_pretrain(config("encoder_mjde"), torch.device("cpu"))
    model = build_pretrain(config(name), torch.device("cpu"))
    core = model.backbone.encoder.core
    assert core.order == order
    assert len(core.spatial) == len(core.temporal) == 6
    assert not hasattr(core, "fusion_gates")
    baseline_core = baseline.backbone.encoder.core
    expected = sum(p.numel() for p in baseline_core.parameters()) - baseline_core.fusion_gates.numel()
    assert sum(p.numel() for p in core.parameters()) == expected


@pytest.mark.parametrize('order', ['s2t', 't2s'])
def test_three_stage_single_path_keeps_only_requested_blocks(order):
    baseline = build_pretrain(config('encoder_mjde'), torch.device('cpu'))
    settings = config('encoder_mjde')
    settings['encoder']['fusion_gate'] = 'static_feature'
    settings['ablation']['encoder'] = 'mjde_' + order + '3'
    model = build_pretrain(settings, torch.device('cpu'))
    core = model.backbone.encoder.core
    original = baseline.backbone.encoder.core
    assert core.order == order
    assert len(core.spatial) == len(core.temporal) == 3
    assert not hasattr(core, 'fusion_gates')
    source = ((original.s2t_spatial, original.s2t_temporal) if order == 's2t'
              else (original.t2s_spatial, original.t2s_temporal))
    expected = sum(p.numel() for group in source for p in group.parameters())
    expected += sum(p.numel() for p in original.output_norm.parameters())
    assert sum(p.numel() for p in core.parameters()) == expected
    visible = torch.rand(2, 19, 6) > .4
    with torch.no_grad():
        output = core(torch.randn(2, 19, 6, 200), visible)
    assert output.shape == (2, 19, 6, 200)
    assert torch.all(output[~visible] == 0)


def test_average_mjde_keeps_both_paths_with_no_learnable_gate():
    baseline = build_pretrain(config("encoder_mjde"), torch.device("cpu"))
    model = build_pretrain(config("encoder_mjde_average"), torch.device("cpu"))
    core = model.backbone.encoder.core
    assert len(core.s2t_spatial) == len(core.t2s_temporal) == 3
    assert not hasattr(core, "fusion_gates")
    baseline_core = baseline.backbone.encoder.core
    expected = sum(p.numel() for p in baseline_core.parameters()) - baseline_core.fusion_gates.numel()
    assert sum(p.numel() for p in core.parameters()) == expected


def test_mix1only_keeps_paths_separate_until_final_average():
    baseline = build_pretrain(config("encoder_mjde"), torch.device("cpu"))
    model = build_pretrain(config("encoder_mjde_mix1only"), torch.device("cpu"))
    core = model.backbone.encoder.core
    assert len(core.s2t_spatial) == len(core.s2t_temporal) == 3
    assert len(core.t2s_spatial) == len(core.t2s_temporal) == 3
    assert not hasattr(core, "fusion_gates")
    baseline_core = baseline.backbone.encoder.core
    expected_count = sum(p.numel() for p in baseline_core.parameters()) - baseline_core.fusion_gates.numel()
    assert sum(p.numel() for p in core.parameters()) == expected_count

    core.eval()
    tokens = torch.randn(2, 19, 6, 200)
    visible = torch.rand(2, 19, 6) > .4
    mask = visible.unsqueeze(-1)
    s2t = t2s = tokens * mask
    for stage in range(3):
        s2t = core.s2t_temporal[stage](core.s2t_spatial[stage](s2t, visible), visible) * mask
        t2s = core.t2s_spatial[stage](core.t2s_temporal[stage](t2s, visible), visible) * mask
    expected = core.output_norm((s2t + t2s) * .5 * mask) * mask
    torch.testing.assert_close(core(tokens, visible), expected, rtol=0, atol=0)


def test_invalid_config_is_not_silently_ignored():
    settings = config("encoder_cbramod")
    settings["ablation"]["mask_mode"] = "dense_zero"
    with pytest.raises(ValueError, match="dense_zero"):
        resolve_ablation(settings)
    settings = config("pe_shpe")
    settings["ablation"]["typo"] = True
    with pytest.raises(ValueError, match="Unknown"):
        resolve_ablation(settings)


@pytest.mark.parametrize("dataset", ["bciciv2a", "chb", "faced", "hmc", "isruc", "physio",
                                     "seed-v", "siena", "stress", "tuab", "tuev", "tusz"])
def test_csbrain_downstream_montages_and_short_sequences(dataset):
    from src.data.datasets.registry import get_dataset_spec
    settings = config("encoder_csbrain")
    settings["ablation"]["depth"] = 1
    backbone = build_backbone(settings).eval()
    spec = get_dataset_spec(dataset)
    names = spec.dataset_class.channel_names
    regions, order = region_order(names, dataset)
    from ablation.montage import canonical
    ignored = {canonical(name) for name in getattr(spec.dataset_class, "ignored_channel_names", ())}
    assert sorted(order) == [i for i, name in enumerate(names) if canonical(name) not in ignored]
    backbone.set_channels(names, dataset)
    # T=1 (SEED-V), T=4 (MI), T=6 (window padding), variable C.
    for patches in (1, 4, 6):
        tokens = torch.randn(1, len(names), patches, 200)
        output = backbone.encoder(tokens, torch.ones(tokens.shape[:-1], dtype=torch.bool))
        assert output.shape == tokens.shape
        assert torch.isfinite(output).all()


def test_factory_connection_restored_on_exception():
    from ablation.integration import connected_engine
    from src.training import engine
    before = (engine.PretrainModel, engine.EEGEncoder, engine.build_finetune)
    with pytest.raises(RuntimeError, match="test"):
        with connected_engine():
            assert engine.PretrainModel is not before[0]
            raise RuntimeError("test")
    assert (engine.PretrainModel, engine.EEGEncoder, engine.build_finetune) == before
