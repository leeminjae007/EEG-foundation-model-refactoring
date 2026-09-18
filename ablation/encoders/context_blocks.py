"""Context masks around native block projections, norms, FFNs and residuals."""
import torch
from torch.nn import functional as F


def masked_mha(module, tokens, visible, structure=None):
    batch, length, _ = tokens.shape
    allowed = visible[:, None, :].expand(batch, length, length)
    if structure is not None:
        allowed = allowed & structure[None]
    has_keys = allowed.any(-1)
    safe = allowed.clone()
    # A dummy key prevents NaNs; its entire query output is discarded below.
    safe[..., 0] |= ~has_keys
    blocked = (~safe)[:, None].expand(-1, module.num_heads, -1, -1)
    blocked = blocked.reshape(batch * module.num_heads, length, length)
    output = module(tokens, tokens, tokens, attn_mask=blocked, need_weights=False)[0]
    return output.masked_fill(~(visible & has_keys)[..., None], 0)


def labram_attention(module, tokens, visible):
    batch, length, dim = tokens.shape
    bias = None
    if module.q_bias is not None:
        bias = torch.cat((module.q_bias, torch.zeros_like(module.v_bias), module.v_bias))
    qkv = F.linear(tokens, module.qkv.weight, bias)
    qkv = qkv.reshape(batch, length, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
    query, key, value = qkv.unbind(0)
    if module.q_norm is not None:
        query = module.q_norm(query).type_as(value)
    if module.k_norm is not None:
        key = module.k_norm(key).type_as(value)
    scores = (query * module.scale) @ key.transpose(-2, -1)
    scores = scores.masked_fill(~visible[:, None, None, :], -torch.inf)
    scores = torch.where(visible.any(-1)[:, None, None, None], scores, torch.zeros_like(scores))
    weights = module.attn_drop(scores.softmax(-1) * visible[:, None, None, :])
    output = (weights @ value).transpose(1, 2).reshape(batch, length, dim)
    return module.proj_drop(module.proj(output)).masked_fill(~visible[..., None], 0)


def labram_block(block, tokens, visible):
    attention = labram_attention(block.attn, block.norm1(tokens), visible)
    if block.gamma_1 is not None:
        attention = block.gamma_1 * attention
    tokens = tokens + block.drop_path(attention)
    ff = block.mlp(block.norm2(tokens))
    if block.gamma_2 is not None:
        ff = block.gamma_2 * ff
    return (tokens + block.drop_path(ff)).masked_fill(~visible[..., None], 0)


def cbramod_block(block, tokens, visible):
    batch, channels, patches, dim = tokens.shape
    normalized = block.norm1(tokens)
    spatial = normalized[..., :dim // 2].permute(0, 2, 1, 3).reshape(batch * patches, channels, dim // 2)
    temporal = normalized[..., dim // 2:].reshape(batch * channels, patches, dim // 2)
    spatial_mask = visible.permute(0, 2, 1).reshape(batch * patches, channels)
    temporal_mask = visible.reshape(batch * channels, patches)
    spatial = masked_mha(block.self_attn_s, spatial, spatial_mask)
    temporal = masked_mha(block.self_attn_t, temporal, temporal_mask)
    spatial = spatial.reshape(batch, patches, channels, dim // 2).permute(0, 2, 1, 3)
    temporal = temporal.reshape(batch, channels, patches, dim // 2)
    tokens = tokens + block.dropout1(torch.cat((spatial, temporal), -1))
    return (tokens + block._ff_block(block.norm2(tokens))).masked_fill(~visible[..., None], 0)


def window_attention(block, tokens, visible):
    batch, channels, original_patches, dim = tokens.shape
    window = min(original_patches, 5)
    pad = (-original_patches) % window
    tokens = F.pad(tokens, (0, 0, 0, pad))
    visible = F.pad(visible, (0, pad), value=False)
    patches, windows = original_patches + pad, (original_patches + pad) // window
    tokens = tokens.reshape(batch, channels, windows, window, dim).permute(0, 3, 1, 2, 4)
    mask = visible.reshape(batch, channels, windows, window).permute(0, 3, 1, 2)
    tokens = masked_mha(block.inter_window_attn, tokens.reshape(batch * window * channels, windows, dim),
                        mask.reshape(batch * window * channels, windows))
    tokens = tokens.reshape(batch, window, channels, windows, dim).permute(0, 2, 3, 1, 4)
    tokens = tokens.reshape(batch, channels, patches, dim)[:, :, :original_patches]
    return block.dropout2(tokens)


def region_attention(block, tokens, visible):
    batch, channels, patches, dim = tokens.shape
    pooled = torch.zeros_like(tokens)
    for indices in block.region_indices_dict.values():
        mask = visible[:, indices, :, None]
        region = tokens[:, indices].masked_fill(~mask, 0)
        mean = region.sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1)
        pooled[:, indices] = mean
    enhanced = tokens + block.global_fc(pooled)
    rows = enhanced.permute(0, 2, 1, 3).reshape(batch * patches, channels, dim)
    valid = visible.permute(0, 2, 1).reshape(batch * patches, channels)
    structure = block.region_attn_mask.to(tokens.device).eq(0)
    rows = masked_mha(block.inter_region_attn, rows, valid, structure)
    return block.dropout1(rows.reshape(batch, patches, channels, dim).permute(0, 2, 1, 3))


def csbrain_block(block, tokens, visible):
    tokens = tokens + window_attention(block, block.norm1(tokens), visible)
    tokens = tokens.masked_fill(~visible[..., None], 0)
    tokens = tokens + region_attention(block, block.norm2(tokens), visible)
    return (tokens + block._ff_block(block.norm3(tokens))).masked_fill(~visible[..., None], 0)
