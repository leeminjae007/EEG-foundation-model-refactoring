"""The active pretraining mask uses nearest 3--7 electrodes, never a fixed radius."""

import pytest
import torch
import yaml

from src.modules.geometry_masking import physical_channel_coordinates
from src.modules.masking import make_masks
from src.training.runtime import ROOT


def test_default_pretraining_mask_is_nearest_and_exact_half():
    config = yaml.safe_load((ROOT / "configs/pretrain.yaml").read_text(encoding="utf-8"))
    settings = config["masking"]
    assert settings["spatial_selection"] == "nearest_channels"
    assert (settings["min_channels"], settings["max_channels"]) == (3, 7)
    assert "radius_m" not in settings
    coordinates = physical_channel_coordinates(config["data"]["channel_names"])
    torch.manual_seed(42)
    masks = make_masks(2, len(coordinates), 30, settings, torch.device("cpu"), coordinates)
    expected = round(len(coordinates) * 30 * settings["mask_ratio"])
    assert torch.equal(masks["target_mask"], ~masks["context_mask"])
    assert (masks["target_mask"].sum((1, 2)) == expected).all()
    assert "geometry_mean_nearest_extent_m" in masks["masking_diagnostics"]


def test_fixed_metric_radius_is_rejected():
    config = yaml.safe_load((ROOT / "configs/pretrain.yaml").read_text(encoding="utf-8"))
    settings = dict(config["masking"], radius_m=0.03)
    coordinates = physical_channel_coordinates(config["data"]["channel_names"])
    with pytest.raises(ValueError, match="fixed-metric-radius"):
        make_masks(1, len(coordinates), 30, settings, torch.device("cpu"), coordinates)
