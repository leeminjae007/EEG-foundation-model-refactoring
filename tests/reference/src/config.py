"""Configuration loading for the canonical and final EEG-MAE models."""

import copy
import json
import os
from pathlib import Path

import yaml


PRETRAIN_SECTIONS = (
    "experiment", "data", "patch_encoder", "latent_tokenizer", "encoder",
    "position", "masking", "mae", "optimization", "runtime", "tracking",
)
DOWNSTREAM_SECTIONS = (
    "experiment", "data", "model", "optimization", "runtime", "tracking",
    "lineage",
)
DOWNSTREAM_DATASETS = {
    "bciciv2a", "chb", "faced", "hmc", "isruc", "mumtaz", "physio", "seed-v",
    "seed-vig", "siena", "speech", "stress", "tuab", "tuev",
}
OUTPUT_ROOTS = (
    Path("/gpfs/data/oermannlab/users/ml10266/workspace/eeg-foundation-model/outputs"),
    Path("/gpfs/data/oermannlab/users/ml10266/workspace/ijepa/outputs"),
)
SEEDVIG_ROOT = Path("/gpfs/data/oermannlab/users/ml10266/Data/eeg_foundation_downstream")
SEEDVIG_PATH = Path("SEED-VIG/processed_cbramod_subject_split")


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _load(path):
    with Path(path).open(encoding="utf-8") as source:
        return yaml.safe_load(source)


def _override_output_root(config):
    override = os.environ.get("EEGFM_OUTPUT_ROOT")
    if not override:
        return
    output = Path(config["runtime"]["output_dir"])
    for root in OUTPUT_ROOTS:
        try:
            relative = output.relative_to(root)
        except ValueError:
            continue
        config["runtime"]["output_dir"] = str(Path(override) / relative)
        return


def _validate_seedvig(data):
    _check(Path(data["root"]) == SEEDVIG_ROOT, "incorrect SEED-VIG root")
    _check(
        Path(data["processed_path"]) == SEEDVIG_PATH,
        "SEED-VIG must use the GR5 subject-disjoint split",
    )
    manifest_path = SEEDVIG_ROOT / SEEDVIG_PATH / "split_manifest.json"
    _check(manifest_path.is_file(), "SEED-VIG split manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "train_subject_ids": list(range(1, 14)),
        "val_subject_ids": list(range(14, 18)),
        "test_subject_ids": list(range(18, 22)),
        "sample_counts": {"train": 13275, "val": 3540, "test": 3540},
    }
    _check(
        manifest.get("protocol_id") == "gr5_subject_disjoint",
        "incorrect SEED-VIG protocol",
    )
    for key, value in expected.items():
        _check(manifest.get(key) == value, f"incorrect SEED-VIG {key}")


def load_pretrain_config(path):
    config = _load(path)
    _override_output_root(config)
    return resolve_pretrain_config(config)


def resolve_pretrain_config(config):
    config = copy.deepcopy(config)
    _check(tuple(config) == PRETRAIN_SECTIONS, "incorrect pretraining sections")
    profile = config["experiment"].get("profile", "canonical")
    _check(
        profile in {
            "canonical",
            "canonical_stagewise_drop_path",
            "historical_d192_reproduction",
        },
        "unsupported pretraining profile",
    )
    historical_reproduction = profile == "historical_d192_reproduction"
    stagewise_drop_path = profile == "canonical_stagewise_drop_path"
    data = config["data"]
    patch = config["patch_encoder"]
    encoder = config["encoder"]
    position = config["position"]
    masking = config["masking"]
    mae = config["mae"]
    optimization = config["optimization"]
    runtime = config["runtime"]
    tracking = config["tracking"]

    segment_samples = round(data["sample_rate"] * data["segment_seconds"])
    patch_samples = round(data["sample_rate"] * data["patch_seconds"])
    _check(segment_samples % patch_samples == 0, "segment must fit full patches")
    _check(patch["patch_samples"] == patch_samples, "patch size mismatch")
    _check(
        len(patch["time_conv_channels"])
        == len(patch["time_kernel_sizes"])
        == len(patch["time_strides"]),
        "temporal convolution lists must match",
    )
    _check(
        patch["frequency_bins"] == patch_samples // 2 + 1,
        "rFFT bin count mismatch",
    )
    tokenizer_architecture = patch.get("architecture", "project_dual_concat")
    _check(
        tokenizer_architecture in {"project_dual_concat", "cbramod"},
        "unsupported tokenizer architecture",
    )
    tokenizer_fusion = patch["fusion"]
    _check(
        patch.get("frequency_input_transform", "log1p_magnitude")
        in {"log1p_magnitude", "magnitude"},
        "unsupported frequency input transform",
    )
    _check(
        patch.get("frequency_projection", "mlp")
        in {"mlp", "linear_dropout"},
        "unsupported frequency projection",
    )
    if tokenizer_architecture == "cbramod":
        _check(
            patch["patch_samples"] == 200
            and patch["embed_dim"] == 200
            and patch["time_conv_channels"] == [25, 25, 25]
            and patch["time_kernel_sizes"] == [49, 3, 3]
            and patch["time_strides"] == [25, 1, 1]
            and patch["time_group_norm_groups"] == 5
            and patch["frequency_bins"] == 101
            and patch["time_feature_dim"] == 200
            and patch["frequency_feature_dim"] == 200
            and patch["fusion"] == "full_add"
            and patch.get("frequency_input_transform") == "magnitude"
            and patch.get("frequency_projection") == "linear_dropout",
            "CBraMod tokenizer settings must match the official time/frequency stem",
        )
    _check(
        tokenizer_fusion in {"concat", "concat_mixer", "full_add"},
        "unsupported tokenizer fusion",
    )
    if tokenizer_fusion in {"concat", "concat_mixer"}:
        _check(
            patch["time_feature_dim"] + patch["frequency_feature_dim"]
            == patch["embed_dim"],
            "tokenizer feature dimensions must sum to embed_dim",
        )
    else:
        _check(
            patch["time_feature_dim"] == patch["embed_dim"]
            and patch["frequency_feature_dim"] == patch["embed_dim"],
            "full-add tokenizer branches must both use embed_dim",
        )
    _check(
        config["latent_tokenizer"] == {"mode": "none"},
        "region compression is not used",
    )
    _check(
        patch["embed_dim"] == encoder["embed_dim"],
        "tokenizer and encoder dimensions must match",
    )

    architecture = encoder.setdefault("architecture", "s2t_t2s")
    _check(architecture == "s2t_t2s", "only canonical s2t_t2s is retained")
    encoder.setdefault("norm_type", "rms_norm")
    encoder.setdefault("activation", "gelu")
    for heads in (encoder["spatial_heads"], encoder["temporal_heads"]):
        _check(encoder["embed_dim"] % heads == 0, "invalid attention heads")
    expected_norm = "layer_norm" if historical_reproduction else "rms_norm"
    _check(encoder["norm_type"] == expected_norm, "incorrect profile normalization")
    _check(encoder["activation"] == "gelu", "canonical best uses GELU")
    if historical_reproduction:
        _check(
            encoder["weight_initialization"] == "pytorch_default",
            "historical reproduction requires the original initialization",
        )
        _check(
            patch == {
                "patch_samples": 200,
                "embed_dim": 192,
                "time_conv_channels": [64, 128, 192],
                "time_kernel_sizes": [15, 7, 3],
                "time_strides": [5, 3, 2],
                "time_group_norm_groups": 8,
                "time_feature_dim": 96,
                "frequency_bins": 101,
                "frequency_hidden_dim": 96,
                "frequency_feature_dim": 96,
                "fusion": "concat",
                "dropout": 0.1,
            },
            "historical reproduction tokenizer differs from the archived run",
        )
        _check(
            encoder == {
                "architecture": "s2t_t2s",
                "embed_dim": 192,
                "depth": 12,
                "spatial_heads": 4,
                "temporal_heads": 4,
                "mlp_ratio": 4.0,
                "dropout": 0.1,
                "attention_dropout": 0.0,
                "drop_path_rate": 0.1,
                "norm_epsilon": 1e-5,
                "norm_type": "layer_norm",
                "activation": "gelu",
                "init_std": 0.02,
                "weight_initialization": "pytorch_default",
                "fusion": "gated_sum",
                "fusion_schedule": "stagewise",
                "hierarchical_summary": False,
            },
            "historical reproduction encoder differs from the archived run",
        )
    else:
        _check(
            encoder["weight_initialization"] == "kaiming_normal_fan_out_relu",
            "canonical pretraining uses CBraMod Kaiming initialization",
        )
    _check(encoder["depth"] == 12, "canonical best uses depth 12")
    _check(
        encoder["fusion"] == "gated_sum"
        and encoder["fusion_schedule"] == "stagewise",
        "canonical best uses stage-wise gated fusion",
    )

    _check(
        position["spatial"] == "spherical_harmonic_factorized"
        and position["spatial_projection"] == "learnable_linear",
        "only factorized SH-linear PE is retained",
    )
    position_fusion = position.get("fusion", "concat")
    component_normalization = position.get("component_normalization", "none")
    component_scaling = position.get("component_scaling", "none")
    post_fusion_activation = position.get("post_fusion_activation", "none")
    post_fusion_normalization = position.get("post_fusion_normalization", "none")
    spatial_attention_sh_bias = position.get("spatial_attention_sh_bias", False)
    temporal_attention_sincos_bias = position.get(
        "temporal_attention_sincos_bias", False
    )
    injection = position.get('encoder_injection', 'additive')
    _check(injection in {'additive', 'attention_bias'}, 'bad encoder PE injection')
    if injection == 'attention_bias':
        _check(position_fusion == 'concat', 'absolute PE bias requires concat PE')
        _check(not spatial_attention_sh_bias and not temporal_attention_sincos_bias,
               'absolute PE replacement cannot also enable the GR9 relative biases')
    _check(
        position_fusion in {"concat", "concat_mixer", "full_add"},
        "unsupported position fusion",
    )
    _check(
        component_normalization in {"none", "rms"},
        "bad position component normalization",
    )
    _check(
        component_scaling in {"none", "learnable"},
        "bad position component scaling",
    )
    _check(
        post_fusion_activation in {"none", "gelu"},
        "bad post-fusion position activation",
    )
    _check(
        post_fusion_normalization in {"none", "rms_norm"},
        "bad post-fusion position normalization",
    )
    _check(
        isinstance(spatial_attention_sh_bias, bool),
        "spatial_attention_sh_bias must be boolean",
    )
    _check(
        isinstance(temporal_attention_sincos_bias, bool),
        "temporal_attention_sincos_bias must be boolean",
    )
    if position_fusion in {"concat", "concat_mixer"}:
        _check(
            position["encoder_spatial_dim"] + position["encoder_temporal_dim"]
            == encoder["embed_dim"],
            "encoder PE dimensions must sum to D",
        )
        _check(
            position["decoder_spatial_dim"] + position["decoder_temporal_dim"]
            == mae["decoder_dim"],
            "decoder PE dimensions must sum to decoder_dim",
        )
    else:
        _check(
            position["encoder_spatial_dim"] == encoder["embed_dim"]
            and position["encoder_temporal_dim"] == encoder["embed_dim"],
            "full-add encoder PE branches must both use D",
        )
        _check(
            position["decoder_spatial_dim"] == mae["decoder_dim"]
            and position["decoder_temporal_dim"] == mae["decoder_dim"],
            "full-add decoder PE branches must both use decoder_dim",
        )
    if historical_reproduction:
        _check(
            position == {
                "spatial": "spherical_harmonic_factorized",
                "spatial_pe_type": "spherical_harmonic",
                "spherical_harmonic_max_degree": 4,
                "spherical_harmonic_orders": "all",
                "spatial_projection": "learnable_linear",
                "fusion": "concat",
                "component_normalization": "none",
                "encoder_spatial_dim": 144,
                "encoder_temporal_dim": 48,
                "decoder_spatial_dim": 72,
                "decoder_temporal_dim": 24,
                "fixed_projection_seed": 42,
                "spatial_coordinate_scale": 2 * 3.141592653589793,
                "temporal": "sinusoidal",
                "temporal_max_period": 10000.0,
            },
            "historical reproduction SHPE differs from the archived run",
        )

    masking_policy = masking.get("policy")
    if masking_policy == "geometry_tubelet":
        _check(
            isinstance(masking.get("mask_ratio"), (int, float))
            and 0 < masking["mask_ratio"] < 1,
            "geometry mask ratio must satisfy 0 < mask_ratio < 1",
        )
        _check(
            set(masking) == {
                "policy",
                "mask_ratio",
                "min_radius_degrees",
                "max_radius_degrees",
                "min_time_patches",
                "max_time_patches",
            },
            "geometry_tubelet masking has missing or unknown settings",
        )
        _check(
            0
            < masking["min_radius_degrees"]
            <= masking["max_radius_degrees"]
            < 180,
            "geometry radii must satisfy 0 < min <= max < 180 degrees",
        )
        _check(
            1
            <= masking["min_time_patches"]
            <= masking["max_time_patches"]
            <= segment_samples // patch_samples,
            "geometry time spans must fit the patch grid",
        )
    elif masking_policy == "leiden_multiscale":
        _check(
            set(masking) == {
                "policy",
                "mask_ratio",
                "max_mask_ratio",
                "time_ranges",
                "graph_neighbors",
                "resolution_values",
                "distance_scale_degrees",
                "partition_seed",
                "max_block_attempts",
                "max_sample_restarts",
                "allow_overlap",
            },
            "leiden_multiscale masking has missing or unknown settings",
        )
        _check(
            masking["mask_ratio"] == 0.5,
            "Leiden multiscale mask ratio must be 0.5",
        )
        _check(
            0.5 <= masking["max_mask_ratio"] <= 0.51,
            "Leiden maximum mask ratio must be between 0.5 and 0.51",
        )
        _check(
            masking["time_ranges"] == [[1, 4], [5, 8], [9, 12]],
            "Leiden time ranges must be 1-4, 5-8, and 9-12",
        )
        _check(
            isinstance(masking["graph_neighbors"], int)
            and 0 < masking["graph_neighbors"] < len(data["channel_names"]),
            "invalid Leiden geometry graph neighbor count",
        )
        _check(
            isinstance(masking["resolution_values"], list)
            and masking["resolution_values"]
            and all(
                isinstance(value, (int, float)) and value > 0
                for value in masking["resolution_values"]
            ),
            "invalid Leiden resolutions",
        )
        _check(
            isinstance(masking["distance_scale_degrees"], (int, float))
            and masking["distance_scale_degrees"] > 0,
            "invalid Leiden distance scale",
        )
        _check(
            isinstance(masking["partition_seed"], int)
            and masking["partition_seed"] >= 0,
            "invalid Leiden partition seed",
        )
        _check(
            isinstance(masking["max_block_attempts"], int)
            and masking["max_block_attempts"] > 0
            and isinstance(masking["max_sample_restarts"], int)
            and masking["max_sample_restarts"] > 0,
            "invalid Leiden sampling attempt limits",
        )
        _check(
            masking["allow_overlap"] is False,
            "Leiden community blocks must not overlap",
        )
    elif masking_policy == "random_patch":
        _check(masking.get("mask_ratio") == 0.5, "mask ratio must be exactly 0.5")
        _check(
            masking == {"policy": "random_patch", "mask_ratio": 0.5},
            "legacy random_patch masking accepts only mask_ratio 0.5",
        )
    elif masking_policy == "ijepa_multiblock":
        _check(
            masking == {
                "policy": "ijepa_multiblock",
                "context_scale": [0.85, 1.0],
                "target_scale": [0.15, 0.15],
                "target_aspect_ratio": [0.75, 1.5],
                "num_context_blocks": 1,
                "num_target_blocks": 4,
                "min_context_tokens": 10,
                "allow_overlap": False,
            },
            "I-JEPA multiblock masking must match the archived policy",
        )
    else:
        raise ValueError(f"unsupported masking policy: {masking_policy}")
    _check(mae["objective"] == "raw_patch_mae", "bad MAE objective")
    mae.setdefault("smooth_l1_beta", 0.1)
    mae.setdefault("frequency_loss_weight", 0.0)
    mae.setdefault("phase_loss_weight", 0.0)
    mae.setdefault("spectral_epsilon", 1e-8)
    _check(
        mae["loss"] in {"mse", "l1", "smooth_l1"}
        and not mae["normalize_targets"]
        and not mae["raw_patch_normalize"],
        "raw patch targets require an unnormalized supported loss",
    )
    _check(mae["smooth_l1_beta"] > 0, "Smooth L1 beta must be positive")
    _check(
        mae["frequency_loss_weight"] >= 0
        and mae["phase_loss_weight"] >= 0,
        "spectral loss weights must be non-negative",
    )
    _check(mae["spectral_epsilon"] > 0, "spectral epsilon must be positive")
    if mae["frequency_loss_weight"] or mae["phase_loss_weight"]:
        _check(
            mae["loss"] == "smooth_l1",
            "spectral losses require Smooth L1 waveform loss",
        )
    if mae["phase_loss_weight"]:
        _check(
            mae["frequency_loss_weight"] > 0,
            "phase loss requires the frequency loss variant",
        )
    _check(mae["decoder_dim"] % mae["decoder_heads"] == 0, "bad decoder heads")
    if historical_reproduction:
        _check(
            mae == {
                "objective": "raw_patch_mae",
                "decoder_dim": 96,
                "decoder_depth": 4,
                "decoder_heads": 4,
                "decoder_mlp_ratio": 4.0,
                "decoder_dropout": 0.0,
                "qkv_bias": True,
                "attention_dropout": 0.0,
                "norm_epsilon": 1e-5,
                "init_std": 0.02,
                "normalize_targets": False,
                "raw_patch_normalize": False,
                "loss": "mse",
                "smooth_l1_beta": 0.1,
                "frequency_loss_weight": 0.0,
                "phase_loss_weight": 0.0,
                "spectral_epsilon": 1e-8,
            },
            "historical reproduction decoder/loss differs from the archived run",
        )

    _check(
        optimization["optimizer"] == "adamw"
        and optimization["scheduler"] == "cosine"
        and optimization["mixed_precision"] == "bf16",
        "unsupported pretraining optimization",
    )
    if historical_reproduction:
        _check(
            optimization["epochs"] == 30
            and optimization["batch_size_per_gpu"] == 64
            and optimization["gradient_accumulation_steps"] == 1
            and optimization["base_learning_rate"] == 3e-4
            and optimization["min_learning_rate"] == 1e-6
            and optimization["start_learning_rate"] == 0.0
            and optimization["weight_decay"] == 5e-2
            and optimization["final_weight_decay"] == 5e-2
            and optimization["adam_betas"] == [0.9, 0.95]
            and optimization["adam_epsilon"] == 1e-8
            and optimization["warmup_ratio"] == 0.05
            and optimization["gradient_clip_norm"] == 1.0,
            "historical reproduction optimizer differs from the archived run",
        )
        _check(
            patch["dropout"] == 0.1
            and encoder["dropout"] == 0.1
            and encoder["attention_dropout"] == 0.0
            and encoder["drop_path_rate"] == 0.1
            and mae["decoder_dropout"] == 0.0
            and mae["attention_dropout"] == 0.0,
            "historical reproduction regularization differs from the archived run",
        )
    else:
        _check(
            optimization["base_learning_rate"] == 5e-4
            and optimization["min_learning_rate"] == 1e-5
            and optimization["weight_decay"] == 5e-2
            and optimization["adam_betas"] == [0.9, 0.999]
            and optimization["adam_epsilon"] == 1e-8
            and optimization["gradient_clip_norm"] == 1.0,
            "pretraining optimizer must match CBraMod",
        )
        _check('warmup_ratio' not in optimization and 'start_learning_rate' not in optimization,
               'use explicit warmup_epochs for the opt-in canonical warmup ablation')
        warmup_epochs = optimization.get('warmup_epochs', 0)
        _check(isinstance(warmup_epochs, (int, float)) and
               0 <= warmup_epochs < optimization['epochs'], 'bad pretrain warmup_epochs')
        _check(
            patch["dropout"] == 0.1
            and encoder["dropout"] == 0.1
            and encoder["attention_dropout"] == 0.1
            and encoder["drop_path_rate"] == (0.1 if stagewise_drop_path else 0.0)
            and mae["decoder_dropout"] == 0.1
            and mae["attention_dropout"] == 0.1,
            "pretraining regularization does not match the selected canonical profile",
        )
    _check(runtime["num_gpus"] == 4, "pretraining requires four GPUs")
    _check(
        runtime["num_nodes"] * runtime["gpus_per_node"] == 4,
        "invalid distributed topology",
    )
    expected_project = (
        "eeg_jepa_pretraining"
        if historical_reproduction else "eeg_mae_pretraining"
    )
    _check(tracking["project"] == expected_project, "bad W&B project")
    _check(tracking["group"] == config["experiment"]["trial_name"], "bad group")

    config["resolved"] = {
        "dataset_dir": str(Path(data["root"]) / data["processed_path"]),
        "segment_samples": segment_samples,
        "num_patches": segment_samples // patch_samples,
        "effective_global_batch_size": (
            optimization["batch_size_per_gpu"]
            * runtime["num_gpus"]
            * optimization["gradient_accumulation_steps"]
        ),
    }
    return config


def load_downstream_config(path):
    config = _load(path)
    _override_output_root(config)
    _check(tuple(config) == DOWNSTREAM_SECTIONS, "incorrect downstream sections")
    data = config["data"]
    model = config["model"]
    optimization = config["optimization"]
    runtime = config["runtime"]
    tracking = config["tracking"]
    lineage = config["lineage"]

    classification_loss = optimization.get("classification_loss", "ce")
    _check(
        classification_loss in {
            "ce", "weighted_ce", "focal_ce", "class_balanced_ce",
            "balanced_softmax",
        },
        "unsupported classification loss",
    )
    if classification_loss in {
        "weighted_ce", "class_balanced_ce", "balanced_softmax",
    }:
        counts = optimization.get("class_counts")
        _check(
            isinstance(counts, list)
            and len(counts) >= 2
            and all(isinstance(value, int) and value > 0 for value in counts),
            "class-prior losses require positive integer class_counts",
        )
    if classification_loss == "focal_ce":
        gamma = optimization.get("focal_gamma", 2.0)
        _check(isinstance(gamma, (int, float)) and gamma >= 0, "bad focal_gamma")
    if classification_loss == "class_balanced_ce":
        beta = optimization.get("class_balance_beta", 0.9999)
        _check(isinstance(beta, (int, float)) and 0 <= beta < 1,
               "bad class_balance_beta")

    _check(data["dataset"] in DOWNSTREAM_DATASETS, "unsupported dataset")
    if data["dataset"] == "seed-vig":
        _validate_seedvig(data)
        _check(
            optimization["selection_metric"] == "r2",
            "SEED-VIG must select validation R2",
        )
    else:
        _check(
            optimization["selection_metric"] == "balanced_accuracy",
            "classification tasks must select validation balanced accuracy",
        )
    _check(model["type"] == "eeg_mae", "only EEG-MAE downstream is retained")
    _check(model.get("transfer_mode", "full") in {"full", "frozen", "random"},
           "unsupported transfer control")
    if data["dataset"] == "isruc":
        _check(model.get("transfer_mode", "full") == "full",
               "ISRUC transfer controls are not implemented")
    _check(model["checkpoint"] == lineage["source_checkpoint"], "bad lineage")
    model.setdefault("pooling", "all_patch_reps")
    _check(model["pooling"] == "all_patch_reps",
           "only all_patch_reps pooling is retained")
    expected_head_activation = "gelu"
    _check(
        model.get("head_activation", "gelu") == expected_head_activation,
        f"{data['dataset']} requires {expected_head_activation.upper()} head activation",
    )
    _check(
        optimization["optimizer"] == "adamw"
        and optimization["scheduler"] == "cosine"
        and optimization["mixed_precision"] == "bf16",
        "unsupported downstream optimization",
    )
    _check(not optimization.get("warmup_ratio", 0.0), "use explicit warmup_epochs")
    warmup_epochs = optimization.get("warmup_epochs", 0)
    _check(isinstance(warmup_epochs, (int, float))
           and 0 <= warmup_epochs < optimization["epochs"], "bad warmup_epochs")
    metrics = {
        "balanced_accuracy", "weighted_f1", "kappa", "auroc", "auprc",
        "pearson", "r2", "rmse",
    }
    selected = [
        optimization["selection_metric"],
        *optimization.get("comparison_selection_metrics", []),
    ]
    _check(set(selected) <= metrics and len(selected) == len(set(selected)), "bad metrics")
    _check(runtime["num_gpus"] == 1 and runtime["distributed"], "bad runtime")
    subject_cv = data.get('subject_cv')
    if subject_cv is not None:
        _check(data['dataset'] in {'stress', 'bciciv2a'}, 'unsupported subject CV dataset')
        _check(subject_cv.get('method') == 'loso_train_only', 'unsupported subject CV')
        _check(isinstance(subject_cv.get('held_out_subject'), str), 'missing held-out subject')
        _check(not runtime['evaluate_test'], 'LOSO folds cannot evaluate test')
        _check(optimization.get('early_stopping_patience') is None, 'CV needs complete epoch curves')
    fixed_epoch = optimization.get('cv_selected_epoch')
    if fixed_epoch is not None:
        _check(subject_cv is None, 'refit must use the original training split')
        _check(isinstance(fixed_epoch, int) and 1 <= fixed_epoch <= optimization['epochs'],
               'invalid CV selected epoch')
        _check(optimization.get('stop_after_epoch') == fixed_epoch, 'refit must stop at selected epoch')
        _check(bool(optimization.get('cv_selection_report')), 'refit needs selection provenance')
        _check(optimization.get('early_stopping_patience') is None, 'refit epoch is precommitted')
        _check(not optimization.get('comparison_selection_metrics')
               and optimization.get('comparison_selection_metric') is None, 'CV refit has one selector')
    else:
        _check('stop_after_epoch' not in optimization, 'stop_after_epoch requires CV selection')
    _check(
        tracking["project"] == f"downstreamtask_{data['dataset']}",
        "bad downstream W&B project",
    )

    config["resolved"] = {
        "dataset_dir": str(Path(data["root"]) / data["processed_path"]),
        "effective_global_batch_size": (
            optimization["batch_size_per_gpu"]
            * optimization["gradient_accumulation_steps"]
        ),
    }
    return config
