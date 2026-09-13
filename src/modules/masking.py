"""True는 선택된 token이다. Geometry는 합집합, 기존 I-JEPA는 블록별 target을 반환한다."""

import math
import torch

from src.modules.geometry_masking import GeometryTubeletMaskingPolicy


def block_size(channels, patches, scale, aspect, device):
    ratio = torch.empty((), device=device).uniform_(*scale).item()
    aspect_ratio = torch.empty((), device=device).uniform_(*aspect).item()
    area = channels * patches * ratio
    height = int(round(math.sqrt(area * aspect_ratio)))
    width = int(round(math.sqrt(area / aspect_ratio)))
    return max(1, min(height, channels - 1)), max(1, min(width, patches - 1))


def rectangle(channels, patches, height, width, device):
    top = int(torch.randint(0, channels - height + 1, (), device=device).item())
    left = int(torch.randint(0, patches - width + 1, (), device=device).item())
    mask = torch.zeros(channels, patches, dtype=torch.bool, device=device)
    mask[top:top + height, left:left + width] = True
    return mask


def make_masks(batch, channels, patches, config, device, channel_coordinates=None):
    policy = config.get("policy", "ijepa_multiblock")
    if policy == "geometry_tubelet":
        geometry = GeometryTubeletMaskingPolicy(
            config["mask_ratio"], config.get("min_radius_degrees"), config.get("max_radius_degrees"),
            config["min_time_patches"], config["max_time_patches"],
            distance_metric=config.get("distance_metric", "geodesic"), radius_m=config.get("radius_m"))
        valid = torch.ones(batch, channels, patches, dtype=torch.bool, device=device)
        return geometry(batch, channels, patches, valid, channel_coordinates)
    if policy != "ijepa_multiblock":
        raise ValueError("unsupported masking policy: " + str(policy))

    target_size = block_size(channels, patches, config["target_scale"],
                             config["target_aspect_ratio"], device)
    aspect = (channels / patches, channels / patches)
    context_size = block_size(channels, patches, config["context_scale"], aspect, device)
    targets = []
    contexts = []
    for _ in range(batch):
        blocks = []
        for _ in range(config["num_target_blocks"]):
            blocks.append(rectangle(channels, patches, *target_size, device))
        blocks = torch.stack(blocks)
        target_union = blocks.any(dim=0)
        best_count = -1
        for _ in range(20):
            candidate = rectangle(channels, patches, *context_size, device)
            candidate = candidate & ~target_union
            count = int(candidate.sum())
            if count > best_count:
                best_context = candidate
                best_count = count
            if count >= config["min_context_tokens"]:
                break
        assert best_count >= config["min_context_tokens"]
        targets.append(blocks)
        contexts.append(best_context)
    minimum = channels * patches
    for context in contexts:
        minimum = min(minimum, int(context.sum()))
    trimmed = []
    for context in contexts:
        indices = context.flatten().nonzero(as_tuple=False).flatten()[:minimum]
        mask = torch.zeros_like(context).flatten()
        mask[indices] = True
        trimmed.append(mask.reshape_as(context))
    target_blocks = torch.stack(targets)
    counts = target_blocks.flatten(2).sum(dim=2)
    width = int(counts.max())
    target_valid = torch.arange(width, device=device)[None, None] < counts[:, :, None]
    return {"context_mask": torch.stack(trimmed), "target_blocks": target_blocks,
            "target_mask": target_blocks.any(dim=1), "target_token_valid": target_valid}


def gather_targets(grid, blocks):
    batch, channels, patches, dim = grid.shape
    flat_blocks = blocks.flatten(2)
    count = int(flat_blocks.sum(dim=2).max())
    indices = torch.arange(channels * patches, device=grid.device)
    indices = indices.view(1, 1, -1).expand_as(flat_blocks)
    indices = indices.masked_fill(~flat_blocks, channels * patches)
    indices = indices.sort(dim=2).values[:, :, :count]
    grid = grid.flatten(1, 2)[:, None].expand(-1, blocks.shape[1], -1, -1)
    gathered = torch.gather(grid, 2, indices.unsqueeze(-1).expand(-1, -1, -1, dim))
    return gathered.reshape(batch, blocks.shape[1] * count, dim)


def gather_context(grid, context_mask):
    count = int(context_mask[0].sum())
    selected = grid.flatten(1, 2)[context_mask.flatten(1)]
    return selected.reshape(grid.shape[0], count, grid.shape[-1])
