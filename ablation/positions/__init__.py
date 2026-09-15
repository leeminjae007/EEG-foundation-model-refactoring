"""Position embedding alternatives and a shared call interface."""

POSITIONS = ("none", "channel_id", "acpe", "reve4d", "shpe")


def build_position(original, config, location):
    name = config["ablation"]["position"]
    if name == "shpe":
        return original
    from ablation.positions.modules import NoPosition, ChannelID, ACPE, REVE4D
    constructors = {"none": NoPosition, "channel_id": ChannelID, "acpe": ACPE, "reve4d": REVE4D}
    return constructors[name](config, location)


def position_values(module, tokens, coordinates, visible):
    if getattr(module, "uses_tokens", False):
        return module(tokens, coordinates, visible)
    return module(coordinates, tokens.shape[0], tokens.shape[2], tokens.dtype)
