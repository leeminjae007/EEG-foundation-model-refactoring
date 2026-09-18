import copy

import pytest
import torch
import yaml

from src.model import PretrainModel
from src.modules.masking import make_masks
from src.training.checkpoint import load_checkpoint, save_checkpoint, rng_state
from src.training.pretrain_rng import initialize_pretrain_rng, pretrain_rng_mode
from src.training.runtime import ROOT


def test_gr2_config_keeps_full_model_and_independent_masks():
    base = yaml.safe_load((ROOT / "configs/pretrain.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / "configs/pretrain_gr2_geometry.yaml").read_text(encoding="utf-8"))
    for section in ("patch_encoder", "encoder", "position", "mae", "optimization", "seed", "data"):
        assert config[section] == base[section]
    torch.manual_seed(42)
    model = PretrainModel(base, torch.device("cpu"))
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    torch.manual_seed(42)
    restored = PretrainModel(config, torch.device("cpu"))
    for key, value in restored.state_dict().items():
        assert torch.equal(value, weights[key])
    masks = []
    for rank in range(4):
        initialize_pretrain_rng(config, rank)
        mask = make_masks(4, 19, 30, config["masking"], torch.device("cpu"), restored.mask_coordinates)
        assert (mask["target_mask"].sum((1, 2)) == 285).all()
        masks.append(mask["target_mask"])
    assert all(not torch.equal(masks[i], masks[j]) for i in range(4) for j in range(i))
    before = torch.get_rng_state()
    initialize_pretrain_rng(base, 3)
    assert torch.equal(before, torch.get_rng_state())


def test_resume_restores_rank_rng_and_rejects_shared_checkpoint(tmp_path):
    config = dict(seed=42, runtime=dict(pretrain_rank_rng="independent"))
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters())
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 100)
    states, expected = [], []
    for rank in range(4):
        initialize_pretrain_rng(config, rank)
        torch.rand(13)
        states.append(rng_state(torch.device("cpu")))
        expected.append(torch.rand(20))
    path = tmp_path / "last.pth"
    save_checkpoint(path, model, opt, sched, 1, config, states, {"step": 10})
    for rank in range(4):
        initialize_pretrain_rng(config, rank)
        load_checkpoint(path, model, opt, sched, torch.device("cpu"), rank,
                        expected_pretrain_rng=pretrain_rng_mode(config))
        assert torch.equal(torch.rand(20), expected[rank])
    shared = copy.deepcopy(config)
    shared["runtime"].pop("pretrain_rank_rng")
    save_checkpoint(path, model, opt, sched, 1, shared, states, {"step": 10})
    with pytest.raises(ValueError, match="pretrain_rank_rng"):
        load_checkpoint(path, model, opt, sched, torch.device("cpu"), 0, expected_pretrain_rng="independent")


def test_ablation_resume_rejects_rng_policy_change():
    from ablation.pretrain_resume import validate_resume
    cfg = yaml.safe_load((ROOT / "configs/pretrain_gr2_geometry.yaml").read_text(encoding="utf-8"))
    cfg["ablation"] = {}
    saved = dict(config=copy.deepcopy(cfg), epoch=1, extra={}, rng_states=[{}] * 4)
    validate_resume(saved, cfg, 4)
    del saved["config"]["runtime"]["pretrain_rank_rng"]
    with pytest.raises(ValueError, match="pretrain_rank_rng"):
        validate_resume(saved, cfg, 4)
