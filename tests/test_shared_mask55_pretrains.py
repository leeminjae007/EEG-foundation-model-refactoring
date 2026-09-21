from scripts import prepare_shared_mask55_pretrains as shared
from scripts import submit_experiment_downstream as downstream


def test_assignments_and_frozen_recipe():
    configs = shared.resolved_configs()
    assert set(configs) == {"pe-ch_order", "pe-acpe", "pe-4dREVE", "cbramod", "csbrain", "labram"}
    assert {row["account"] for row in configs.values()} == {"hk4935", "yc8820"}
    for slug, row in configs.items():
        config = row["config"]
        assert config["masking"]["policy"] == "geometry_tubelet"
        assert config["masking"]["mask_ratio"] == .55
        assert config["mae"]["decoder_depth"] == 2
        assert config["optimization"]["epochs"] == 40
        assert config["seed"] == 42
        assert config["ablation"]["reference_fusion_gate"] == "patch_feature"
        if slug.startswith("pe-"):
            assert config["encoder"]["fusion_gate"] == "patch_feature"
            assert config["ablation"]["fusion_gate_applicability"] == "active"
        else:
            assert config["ablation"]["depth"] == 12
            assert config["ablation"]["fusion_gate_applicability"] == "not_applicable_encoder_replaced"


def test_exact_three_arms_per_account_and_downstream_is_hk_only():
    assert all(len(arms) == 3 for arms in shared.ASSIGNMENTS.values())
    assert shared.downstream_allowed("hk4935", True)
    assert not shared.downstream_allowed("hk4935", False)
    assert not shared.downstream_allowed("yc8820", False)
    try:
        shared.downstream_allowed("yc8820", True)
    except PermissionError:
        pass
    else:
        raise AssertionError("yc8820 must remain pretrain-only")
    source = open("scripts/prepare_shared_mask55_pretrains.py", encoding="utf-8").read()
    assert '"a100"' in source


def test_attached_downstream_routes_only_to_gl40s():
    for dataset in downstream.DATASETS:
        policy = downstream.resource_policy(shared.ROOT, "mask55-d2-patch-dimension", dataset)
        assert policy["gpu"] == "l40s"
        assert all(part.startswith("gl40s_") for part in policy["partitions"].split(","))
