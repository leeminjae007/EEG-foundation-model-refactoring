from scripts import prepare_shared_mask55_pretrains as shared


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


def test_exact_three_arms_per_account_and_no_downstream_path():
    assert all(len(arms) == 3 for arms in shared.ASSIGNMENTS.values())
    source = open("scripts/prepare_shared_mask55_pretrains.py", encoding="utf-8").read()
    assert "submit_experiment_downstream" not in source
    assert '"a100"' in source
