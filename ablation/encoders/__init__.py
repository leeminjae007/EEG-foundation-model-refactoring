"""Encoder-only adapters: shared tokenizer/PE/decoder live outside these modules."""

from ablation.encoders.mjde import MJDELite, MaskProtocol

ENCODERS = ("labram", "cbramod", "csbrain", "mjde", "mjde_lite")


def build_encoder(original, config):
    settings = config["ablation"]
    name = settings["encoder"]
    if name == "mjde":
        core = original
    elif name == "mjde_lite":
        core = MJDELite(original)
    else:
        from ablation.encoders.paper import PaperEncoder
        core = PaperEncoder(config)
    return MaskProtocol(core, settings["mask_mode"])
