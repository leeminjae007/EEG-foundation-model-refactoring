"""Compose ablations with the existing tokenizer, decoder and training models."""

import torch
from torch import nn

from ablation.encoders import build_encoder
from ablation.positions import build_position, position_values
from src.model import EEGEncoder, PretrainModel
from src.modules.masking import gather_context, gather_targets


class AblationBackbone(nn.Module):
    def __init__(self, original, config):
        super().__init__()
        self.tokenizer = original.tokenizer
        self.encoder = build_encoder(original.encoder, config)
        self.position = build_position(original.position, config, "encoder")
        self.register_buffer("default_channel_coordinates", original.default_channel_coordinates)

    def set_channels(self, names, dataset):
        core = self.encoder.core
        if hasattr(core, "set_channels"):
            core.set_channels(names, dataset)
        if hasattr(self.position, "set_channels"):
            self.position.set_channels(names)

    def forward(self, signals, coordinates, channel_valid, visible):
        signals = signals.masked_fill(~channel_valid[:, :, None], 0.0)
        tokens = self.tokenizer(signals)
        visible = visible & channel_valid[:, :, None]
        positions = position_values(self.position, tokens, coordinates, visible)
        tokens = (tokens + positions) * channel_valid[:, :, None, None]
        return self.encoder(tokens, visible)


class AblationDecoder(nn.Module):
    def __init__(self, original, config):
        super().__init__()
        for name, module in original.named_children():
            self.add_module(name, module)
        self.mask_token = original.mask_token
        self.register_buffer("default_channel_coordinates", original.default_channel_coordinates)
        self.position = build_position(original.position, config, "decoder")

    def forward(self, context_grid, masks):
        batch, channels, patches, _ = context_grid.shape
        projected = self.context_projection(context_grid)
        positions = position_values(self.position, projected, self.default_channel_coordinates,
                                    masks["context_mask"])
        # Same gather / target concatenation / original blocks as src/decoder.py.
        context = gather_context(projected + positions, masks["context_mask"])
        target_positions = gather_targets(positions, masks["target_blocks"])
        blocks = masks["target_blocks"].shape[1]
        target_count = int(masks["target_blocks"][0, 0].sum())
        context_count, width = context.shape[1:]
        targets = target_positions.reshape(batch * blocks, target_count, width) + self.mask_token
        context = context[:, None].expand(-1, blocks, -1, -1).reshape(batch * blocks, context_count, width)
        sequence = torch.cat((context, targets), dim=1)
        for block in self.blocks:
            sequence = block(sequence)
        targets = self.norm(sequence)[:, context_count:]
        prediction = self.output_projection(targets)
        return prediction.reshape(batch, blocks * target_count, prediction.shape[-1])


def build_backbone(config):
    return AblationBackbone(EEGEncoder(config), config)


def build_pretrain(config, device):
    # Construct the common model first so its initialization is identical across
    # arms. Attach paper modules AFTER src's custom initialization has finished.
    model = PretrainModel(config, device)
    model.backbone = AblationBackbone(model.backbone, config).to(device)
    if config["ablation"]["pe_scope"] == "both":
        model.decoder = AblationDecoder(model.decoder, config).to(device)
    return model


def parameter_report(model):
    def count(module):
        return {"total": sum(p.numel() for p in module.parameters()),
                "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad)}
    return {"total": count(model), "tokenizer": count(model.backbone.tokenizer),
            "encoder": count(model.backbone.encoder), "encoder_position": count(model.backbone.position),
            "decoder": count(model.decoder)}
