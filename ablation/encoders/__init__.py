"""Encoder-only adapters: shared tokenizer/PE/decoder live outside these modules."""

from ablation.encoders.mjde import AverageMJDE, Mix1OnlyMJDE, MJDELite, SinglePathMJDE, MaskProtocol

ENCODERS = ("labram", "cbramod", "csbrain", "mjde", "mjde_lite",
            "mjde_s2t6", "mjde_t2s6", "mjde_average", "mjde_mix1only")


def build_encoder(original, config):
    settings = config["ablation"]
    from src.modules.fusion import fusion_gate_mode
    if fusion_gate_mode(config["encoder"]) != "static_feature" and settings["encoder"] != "mjde":
        raise ValueError("Dynamic patch fusion gates require the full MJDE encoder")
    if settings.get("mask_mode", "context_only") != "context_only":
        raise ValueError("Legacy dense_zero checkpoints cannot use context-only encoder blocks")
    name = settings["encoder"]
    if name == "mjde":
        core = original
    elif name == "mjde_lite":
        core = MJDELite(original)
    elif name == "mjde_s2t6":
        core = SinglePathMJDE(original, "s2t")
    elif name == "mjde_t2s6":
        core = SinglePathMJDE(original, "t2s")
    elif name == "mjde_average":
        core = AverageMJDE(original)
    elif name == "mjde_mix1only":
        core = Mix1OnlyMJDE(original)
    else:
        from ablation.encoders.paper import PaperEncoder
        core = PaperEncoder(config)
    return MaskProtocol(core)
