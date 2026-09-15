"""Scope model-factory replacements to the ablation runner's process/call."""

from contextlib import contextmanager
from ablation.models import build_backbone, build_pretrain


@contextmanager
def connected_engine():
    from src.training import engine

    original_pretrain = engine.PretrainModel
    original_encoder = engine.EEGEncoder
    original_finetune = engine.build_finetune

    def build_finetune(config):
        model = original_finetune(config)
        dataset = config["data"]["dataset"]
        spec = engine.get_dataset_spec(dataset)
        model.backbone.set_channels(spec.dataset_class.channel_names, dataset)
        return model

    engine.PretrainModel = build_pretrain
    engine.EEGEncoder = build_backbone
    engine.build_finetune = build_finetune
    try:
        yield engine
    finally:
        engine.PretrainModel = original_pretrain
        engine.EEGEncoder = original_encoder
        engine.build_finetune = original_finetune
