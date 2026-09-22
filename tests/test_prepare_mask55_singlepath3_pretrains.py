import torch

from ablation.models import build_pretrain
from scripts import prepare_mask55_singlepath3_pretrains as campaign


def test_two_frozen_three_stage_paths_are_distinct_pretrains():
    configs = campaign.resolved_configs()
    assert set(configs) == {'s2t3', 't2s3'}
    for slug, config in configs.items():
        assert config['ablation']['encoder'] == campaign.ARMS[slug]
        assert config['masking']['mask_ratio'] == .55
        assert config['mae']['decoder_depth'] == 2
        assert config['encoder']['fusion_gate'] == 'static_feature'
        assert config['ablation']['fusion_gate_applicability'] == 'not_applicable_single_path'
        assert config['optimization']['epochs'] == 40
        model = build_pretrain(config, torch.device('cpu'))
        core = model.backbone.encoder.core
        assert core.order == slug[:3]
        assert len(core.spatial) == len(core.temporal) == 3
        assert len(model.decoder.blocks) == 2
