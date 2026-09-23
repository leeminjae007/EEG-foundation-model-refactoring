import torch

from ablation.models import build_pretrain
from scripts import submit_mask55_owner_four_pretrains as campaign


def test_frozen_owner_four_configurations():
    configs = campaign.resolved_configs()
    assert set(configs) == {"ours-lite", "enc-s2t-6stage", "enc-t2s-6stage", "enc-average-3s"}
    for slug, config in configs.items():
        assert config["ablation"]["encoder"] == campaign.ARMS[slug]
        assert config["masking"]["mask_ratio"] == .55
        assert config["mae"]["decoder_depth"] == 2
        assert config["encoder"]["fusion_gate"] == "static_feature"
        assert config["optimization"]["epochs"] == 40
        model = build_pretrain(config, torch.device("cpu"))
        assert len(model.decoder.blocks) == 2
        core = model.backbone.encoder.core
        if "6stage" in slug:
            assert len(core.spatial) == len(core.temporal) == 6
        elif slug == "ours-lite":
            assert len(core.s2t_spatial) == len(core.t2s_spatial) == 1
        else:
            assert len(core.s2t_spatial) == len(core.t2s_spatial) == 3
