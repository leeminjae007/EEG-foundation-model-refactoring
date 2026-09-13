"""Construction of the retained pretraining encoder and decoder."""

import torch.nn as nn

from src.models.eeg_encoder import build_eeg_encoder
from src.models.reconstruction_decoder import MaskedPatchDecoder
from src.models.transformer_blocks import (
    initialize_historical_transformer,
    rescale_historical_decoder_blocks,
)


def initialize_cbramod_weights(module):
    """Apply CBraMod's Kaiming-normal weight initialization verbatim."""
    if getattr(module, "preserve_identity_initialization", False):
        return
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.kaiming_normal_(
            module.weight, mode="fan_out", nonlinearity="relu"
        )


def build_models(config, device):
    """Build the canonical encoder-decoder pair without training concerns."""
    context_encoder = build_eeg_encoder(config).to(device)
    reconstruction = config["mae"]
    assert reconstruction["objective"] == "raw_patch_mae"
    assert config["latent_tokenizer"]["mode"] == "none"
    decoder = MaskedPatchDecoder(
        embed_dim=config["encoder"]["embed_dim"],
        decoder_dim=reconstruction["decoder_dim"],
        depth=reconstruction["decoder_depth"],
        num_heads=reconstruction["decoder_heads"],
        mlp_ratio=reconstruction["decoder_mlp_ratio"],
        qkv_bias=reconstruction["qkv_bias"],
        dropout=reconstruction["decoder_dropout"],
        attention_dropout=reconstruction["attention_dropout"],
        norm_epsilon=reconstruction["norm_epsilon"],
        norm_type=config["encoder"].get("norm_type", "rms_norm"),
        temporal_max_period=config["position"]["temporal_max_period"],
        default_channel_coordinates=context_encoder.default_channel_coordinates,
        position_type=config["position"]["spatial"],
        spherical_harmonic_max_degree=config["position"].get(
            "spherical_harmonic_max_degree", 4
        ),
        spatial_position_dim=config["position"]["decoder_spatial_dim"],
        temporal_position_dim=config["position"]["decoder_temporal_dim"],
        position_fusion=config["position"].get("fusion", "concat"),
        position_component_normalization=config["position"].get(
            "component_normalization", "none"
        ),
        position_component_scaling=config["position"].get(
            "component_scaling", "none"
        ),
        position_rms_epsilon=config["position"].get("rms_epsilon", 1e-6),
        position_post_fusion_activation=config["position"].get(
            "post_fusion_activation", "none"
        ),
        position_post_fusion_normalization=config["position"].get(
            "post_fusion_normalization", "none"
        ),
        output_dim=config["patch_encoder"]["patch_samples"],
        init_std=(
            reconstruction.get("init_std", 0.02)
            if config["experiment"].get("profile")
            == "historical_d192_reproduction"
            else None
        ),
    )
    initialization = config["encoder"]["weight_initialization"]
    if initialization == "kaiming_normal_fan_out_relu":
        decoder = decoder.to(device)
        context_encoder.apply(initialize_cbramod_weights)
        decoder.apply(initialize_cbramod_weights)
    elif initialization == "pytorch_default":
        # The archived encoder retained module defaults apart from its SH-linear
        # projection. Its decoder initialized all Linear/LayerNorm modules with
        # the Transformer rule and rescaled residual output projections.
        init_std = float(reconstruction.get("init_std", 0.02))
        decoder.apply(
            lambda module: initialize_historical_transformer(module, init_std)
        )
        nn.init.trunc_normal_(
            decoder.mask_token,
            std=init_std,
        )
        rescale_historical_decoder_blocks(decoder.blocks)
        decoder = decoder.to(device)
    else:
        raise ValueError(f"unsupported weight initialization: {initialization}")
    return context_encoder, decoder
