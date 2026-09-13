"""Stateless masking policies for EEG-MAE pretraining."""

import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F


def _mask_result(
    context_mask,
    target_blocks,
    valid_token_mask,
    masking_diagnostics=None,
    require_partition=True,
):
    target_mask = target_blocks.any(dim=1)
    if (context_mask & target_mask).any():
        raise RuntimeError("context and target masks overlap")
    covered = context_mask | target_mask
    if (covered & ~valid_token_mask).any():
        raise RuntimeError("context and target masks must be valid tokens")
    if require_partition and not torch.equal(covered, valid_token_mask):
        raise RuntimeError("context and target must partition all valid tokens")
    counts = target_blocks.flatten(2).sum(dim=2)
    width = int(counts.max())
    target_token_valid = (
        torch.arange(width, device=target_blocks.device)[None, None]
        < counts[:, :, None]
    )
    result = {
        "context_mask": context_mask,
        "target_mask": target_mask,
        "target_blocks": target_blocks,
        "target_positions": target_mask.clone(),
        "valid_token_mask": valid_token_mask.clone(),
        "observation_mask": context_mask.clone(),
        "target_block_valid": counts.gt(0),
        "target_token_valid": target_token_valid,
    }
    if masking_diagnostics:
        result["masking_diagnostics"] = masking_diagnostics
    return result


def _validate_fixed_grid(batch_size, num_latents, num_patches, valid_token_mask):
    expected = (int(batch_size), int(num_latents), int(num_patches))
    if valid_token_mask.dtype != torch.bool or valid_token_mask.shape != expected:
        raise ValueError(f"valid_token_mask must be boolean {expected}")
    if not valid_token_mask.all():
        raise ValueError("pretraining requires the fixed valid TUEG grid")
    return expected


class RandomPatchMaskingPolicy:
    """Legacy policy that samples individual flattened ``[C,T]`` tokens."""

    def __init__(self, mask_ratio):
        self.mask_ratio = float(mask_ratio)
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be between zero and one")

    def __call__(
        self,
        batch_size,
        num_latents,
        num_patches,
        valid_token_mask,
        generator=None,
        **_,
    ):
        batch, channels, patches = _validate_fixed_grid(
            batch_size, num_latents, num_patches, valid_token_mask
        )
        total = channels * patches
        target_count = round(total * self.mask_ratio)
        if not 0 < target_count < total:
            raise ValueError("mask_ratio leaves no context or target tokens")

        target_indices = torch.rand(
            batch, total, generator=generator, device="cpu"
        ).argsort(dim=1)[:, :target_count].to(valid_token_mask.device)
        target_flat = torch.zeros(
            batch, total, dtype=torch.bool, device=valid_token_mask.device
        )
        target_flat.scatter_(1, target_indices, True)
        target_mask = target_flat.reshape(batch, channels, patches)
        context_mask = valid_token_mask & ~target_mask
        return _mask_result(
            context_mask,
            target_mask[:, None],
            valid_token_mask,
        )


class IJEPAMultiBlockMaskingPolicy:
    """Historical rectangular multi-block policy used by the D192 best model."""

    def __init__(
        self,
        context_scale,
        target_scale,
        target_aspect_ratio,
        num_context_blocks,
        num_target_blocks,
        min_context_tokens,
        allow_overlap,
    ):
        self.context_scale = tuple(float(value) for value in context_scale)
        self.target_scale = tuple(float(value) for value in target_scale)
        self.target_aspect_ratio = tuple(
            float(value) for value in target_aspect_ratio
        )
        self.num_context_blocks = int(num_context_blocks)
        self.num_target_blocks = int(num_target_blocks)
        self.min_context_tokens = int(min_context_tokens)
        self.allow_overlap = bool(allow_overlap)
        for name, bounds in (
            ("context_scale", self.context_scale),
            ("target_scale", self.target_scale),
            ("target_aspect_ratio", self.target_aspect_ratio),
        ):
            if len(bounds) != 2 or not 0 < bounds[0] <= bounds[1]:
                raise ValueError(f"invalid {name}")
        if self.context_scale[1] > 1 or self.target_scale[1] > 1:
            raise ValueError("I-JEPA block scales cannot exceed one")
        if self.num_context_blocks != 1 or self.num_target_blocks < 1:
            raise ValueError("historical I-JEPA masking requires one context block")
        if self.min_context_tokens < 1:
            raise ValueError("min_context_tokens must be positive")

    @staticmethod
    def _block_size(channels, patches, scale, aspect, device):
        ratio = torch.empty((), device=device).uniform_(*scale).item()
        aspect_ratio = torch.empty((), device=device).uniform_(*aspect).item()
        area = channels * patches * ratio
        height = int(round(math.sqrt(area * aspect_ratio)))
        width = int(round(math.sqrt(area / aspect_ratio)))
        return (
            max(1, min(height, channels - 1)),
            max(1, min(width, patches - 1)),
        )

    @staticmethod
    def _rectangle(channels, patches, height, width, device):
        top = int(torch.randint(
            0, channels - height + 1, (), device=device
        ).item())
        left = int(torch.randint(
            0, patches - width + 1, (), device=device
        ).item())
        mask = torch.zeros(channels, patches, dtype=torch.bool, device=device)
        mask[top:top + height, left:left + width] = True
        return mask

    @staticmethod
    def _trim(mask, count):
        indices = mask.flatten().nonzero(as_tuple=False).flatten()[:count]
        trimmed = torch.zeros_like(mask).flatten()
        trimmed[indices] = True
        return trimmed.reshape_as(mask)

    def __call__(
        self,
        batch_size,
        num_latents,
        num_patches,
        valid_token_mask,
        generator=None,
        **_,
    ):
        # The archived implementation sampled from the device-global PyTorch
        # RNG.  Keep the unused argument for the common masking-policy API.
        del generator
        batch, channels, patches = _validate_fixed_grid(
            batch_size, num_latents, num_patches, valid_token_mask
        )
        device = valid_token_mask.device
        target_size = self._block_size(
            channels,
            patches,
            self.target_scale,
            self.target_aspect_ratio,
            device,
        )
        context_size = self._block_size(
            channels,
            patches,
            self.context_scale,
            (channels / patches, channels / patches),
            device,
        )

        target_batches = []
        context_masks = []
        for _ in range(batch):
            target_blocks = torch.stack([
                self._rectangle(
                    channels, patches, *target_size, device
                )
                for _ in range(self.num_target_blocks)
            ])
            target_union = target_blocks.any(dim=0)
            candidates = []
            for _ in range(20):
                candidate = self._rectangle(
                    channels, patches, *context_size, device
                )
                if not self.allow_overlap:
                    candidate = candidate & ~target_union
                candidates.append(candidate)
                if int(candidate.sum()) >= self.min_context_tokens:
                    break
            context = max(candidates, key=lambda value: int(value.sum()))
            if int(context.sum()) < self.min_context_tokens:
                raise RuntimeError("failed to sample a valid I-JEPA context block")
            target_batches.append(target_blocks)
            context_masks.append(context)

        minimum_context = min(int(mask.sum()) for mask in context_masks)
        context_mask = torch.stack([
            self._trim(mask, minimum_context) for mask in context_masks
        ])
        target_blocks = torch.stack(target_batches)
        result = _mask_result(
            context_mask,
            target_blocks,
            valid_token_mask,
            require_partition=False,
        )
        result["masking_diagnostics"] = {
            "ijepa_context_token_count": float(minimum_context),
            "ijepa_target_block_token_count": float(target_blocks[0, 0].sum()),
            "ijepa_target_union_token_count": (
                result["target_mask"].sum((1, 2)).float().mean()
            ),
        }
        return result


class GeometryTubeletMaskingPolicy:
    """Mask unions of local electrode circles and contiguous time intervals.

    A proposal selects a random electrode as its spatial center, samples a
    geodesic radius on the unit scalp sphere, and combines the resulting local
    channel circle with a random contiguous patch interval. Proposals are added
    until the exact requested target count is reached. If the final proposal
    crosses the count boundary, its most central space-time tokens are retained.
    """

    def __init__(
        self,
        mask_ratio,
        min_radius_degrees,
        max_radius_degrees,
        min_time_patches,
        max_time_patches,
    ):
        self.mask_ratio = float(mask_ratio)
        self.min_radius_degrees = float(min_radius_degrees)
        self.max_radius_degrees = float(max_radius_degrees)
        self.min_time_patches = int(min_time_patches)
        self.max_time_patches = int(max_time_patches)
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be between zero and one")
        if not 0.0 < self.min_radius_degrees <= self.max_radius_degrees < 180.0:
            raise ValueError("geometry radii must satisfy 0 < min <= max < 180")
        if not 1 <= self.min_time_patches <= self.max_time_patches:
            raise ValueError("time spans must satisfy 1 <= min <= max")

    @staticmethod
    def _coordinates(channel_coordinates, batch, channels):
        if channel_coordinates is None:
            raise ValueError("geometry_tubelet masking requires channel coordinates")
        coordinates = channel_coordinates.detach().float().cpu()
        if not torch.isfinite(coordinates).all():
            raise ValueError("channel coordinates must be finite")
        norms = coordinates.norm(dim=-1)
        if (norms <= 0).any():
            raise ValueError("channel coordinates must be nonzero")
        if coordinates.ndim == 2:
            if coordinates.shape != (channels, 3):
                raise ValueError(
                    "channel_coordinates must have shape [C,3] or [B,C,3]"
                )
            coordinates = F.normalize(coordinates, dim=-1)
            return coordinates.unsqueeze(0).expand(batch, -1, -1)
        if coordinates.shape != (batch, channels, 3):
            raise ValueError("channel_coordinates must have shape [C,3] or [B,C,3]")
        return F.normalize(coordinates, dim=-1)

    def _sample_one(self, spatial_distances, patches, target_count, rng):
        channels = spatial_distances.shape[0]
        target = np.zeros((channels, patches), dtype=np.bool_)
        tubelet_count = 0
        radius_sum = 0.0
        duration_sum = 0.0
        clipped_count = 0
        max_attempts = channels * patches * 8

        for _ in range(max_attempts):
            remaining = target_count - int(target.sum())
            if remaining == 0:
                break
            center = int(rng.integers(0, channels))
            radius_degrees = float(rng.uniform(
                self.min_radius_degrees, self.max_radius_degrees
            ))
            radius = math.radians(radius_degrees)
            duration = int(rng.integers(
                self.min_time_patches,
                min(self.max_time_patches, patches) + 1,
            ))
            start = int(rng.integers(0, patches - duration + 1))
            stop = start + duration

            spatial_distance = spatial_distances[center]
            selected_channels = spatial_distance <= radius
            selected_channels[center] = True
            proposal = np.zeros_like(target)
            proposal[:, start:stop] = selected_channels[:, None]
            proposal &= ~target
            candidate_count = int(proposal.sum())
            if candidate_count == 0:
                continue

            if candidate_count > remaining:
                indices = np.argwhere(proposal)
                temporal_center = (start + stop - 1) / 2.0
                spatial_score = spatial_distance[indices[:, 0]] / max(radius, 1e-8)
                temporal_scale = max(duration / 2.0, 1.0)
                temporal_score = (
                    np.abs(indices[:, 1] - temporal_center) / temporal_scale
                )
                tie_break = rng.random(candidate_count) * 1e-6
                order = np.argsort(spatial_score + temporal_score + tie_break)
                keep = indices[order[:remaining]]
                proposal.fill(False)
                proposal[keep[:, 0], keep[:, 1]] = True
                clipped_count += 1

            target |= proposal
            tubelet_count += 1
            radius_sum += radius_degrees
            duration_sum += duration
            if int(target.sum()) == target_count:
                break

        if int(target.sum()) != target_count:
            raise RuntimeError("geometry tubelet masking failed to reach target ratio")
        return target, tubelet_count, radius_sum, duration_sum, clipped_count

    def __call__(
        self,
        batch_size,
        num_latents,
        num_patches,
        valid_token_mask,
        channel_coordinates=None,
        generator=None,
        **_,
    ):
        batch, channels, patches = _validate_fixed_grid(
            batch_size, num_latents, num_patches, valid_token_mask
        )
        if self.min_time_patches > patches:
            raise ValueError("minimum tubelet time span exceeds the patch grid")
        coordinates = GeometryTubeletMaskingPolicy._coordinates(
            channel_coordinates, batch, channels
        )
        coordinate_array = coordinates.numpy()
        cosine = np.matmul(coordinate_array, coordinate_array.transpose(0, 2, 1))
        spatial_distances = np.arccos(np.clip(cosine, -1.0, 1.0))
        numpy_seed = int(torch.randint(
            0,
            torch.iinfo(torch.int64).max,
            (),
            generator=generator,
            dtype=torch.int64,
        ).item())
        rng = np.random.default_rng(numpy_seed)
        total = channels * patches
        target_count = round(total * self.mask_ratio)
        if not 0 < target_count < total:
            raise ValueError("mask_ratio leaves no context or target tokens")

        masks = []
        tubelet_counts = []
        mean_radii = []
        mean_durations = []
        clipped_counts = []
        for batch_index in range(batch):
            mask, count, radius_sum, duration_sum, clipped = self._sample_one(
                spatial_distances[batch_index], patches, target_count, rng
            )
            masks.append(mask)
            tubelet_counts.append(count)
            mean_radii.append(radius_sum / count)
            mean_durations.append(duration_sum / count)
            clipped_counts.append(clipped)

        target_mask = torch.from_numpy(np.stack(masks)).to(valid_token_mask.device)
        context_mask = valid_token_mask & ~target_mask
        diagnostics = {
            "geometry_tubelet_count": torch.tensor(tubelet_counts).float().mean(),
            "geometry_mean_radius_degrees": torch.tensor(mean_radii).mean(),
            "geometry_mean_time_span_patches": torch.tensor(mean_durations).mean(),
            "geometry_clipped_tubelet_count": (
                torch.tensor(clipped_counts).float().mean()
            ),
        }
        return _mask_result(
            context_mask,
            target_mask[:, None],
            valid_token_mask,
            masking_diagnostics=diagnostics,
        )


class LeidenMultiScaleMaskingPolicy:
    """Mask non-overlapping spatial communities at three temporal scales.

    A bank of Leiden partitions is computed from the fixed geometry-local
    electrode graph at several resolutions. Each proposal selects a partition,
    a seed community, and a contiguous duration. Raw target values never
    influence mask selection. The final community block is never trimmed.
    """

    def __init__(
        self,
        mask_ratio,
        max_mask_ratio,
        time_ranges,
        graph_neighbors,
        resolution_values,
        distance_scale_degrees,
        partition_seed,
        max_block_attempts,
        max_sample_restarts,
        allow_overlap,
    ):
        self.mask_ratio = float(mask_ratio)
        self.max_mask_ratio = float(max_mask_ratio)
        self.time_ranges = tuple(
            (int(bounds[0]), int(bounds[1])) for bounds in time_ranges
        )
        self.graph_neighbors = int(graph_neighbors)
        self.resolution_values = tuple(float(value) for value in resolution_values)
        self.distance_scale_degrees = float(distance_scale_degrees)
        self.partition_seed = int(partition_seed)
        self.max_block_attempts = int(max_block_attempts)
        self.max_sample_restarts = int(max_sample_restarts)
        self.allow_overlap = bool(allow_overlap)
        self._cached_distance_key = None
        self._cached_partition_bank = None
        self._cached_scale_options = None
        self._partition_cache_hit = False
        if self.mask_ratio != 0.5:
            raise ValueError("Leiden multiscale masking requires mask_ratio 0.5")
        if not self.mask_ratio <= self.max_mask_ratio <= 0.51:
            raise ValueError("Leiden max_mask_ratio must be between 0.5 and 0.51")
        if self.time_ranges != ((1, 4), (5, 8), (9, 12)):
            raise ValueError("Leiden time ranges must be 1-4, 5-8, and 9-12")
        if self.graph_neighbors < 1:
            raise ValueError("graph_neighbors must be positive")
        if not self.resolution_values or any(
            value <= 0 for value in self.resolution_values
        ):
            raise ValueError("Leiden resolutions must be positive")
        if self.distance_scale_degrees <= 0:
            raise ValueError("Leiden distance scale must be positive")
        if self.max_block_attempts < 1 or self.max_sample_restarts < 1:
            raise ValueError("Leiden sampling attempt limits must be positive")
        if self.allow_overlap:
            raise ValueError("Leiden community blocks must not overlap")

    @staticmethod
    def _geometry_edges(spatial_distances, neighbors):
        channels = spatial_distances.shape[0]
        if neighbors >= channels:
            raise ValueError("graph_neighbors must be smaller than channel count")
        edges = set()
        for channel in range(channels):
            nearest = np.argsort(spatial_distances[channel])[1:neighbors + 1]
            for other in nearest:
                edges.add(tuple(sorted((channel, int(other)))))
        return tuple(sorted(edges))

    def _partition_bank(self, spatial_distances):
        try:
            import igraph as ig
        except ImportError as error:
            raise RuntimeError(
                "leiden_multiscale masking requires python-igraph"
            ) from error
        channels = spatial_distances.shape[0]
        edges = self._geometry_edges(spatial_distances, self.graph_neighbors)
        scale = math.radians(self.distance_scale_degrees)
        graph = ig.Graph(n=channels, edges=edges, directed=False)
        graph.es["weight"] = [
            math.exp(-0.5 * (spatial_distances[first, second] / scale) ** 2)
            for first, second in edges
        ]
        bank = []
        for resolution in self.resolution_values:
            partition = graph.community_leiden(
                objective_function="modularity",
                weights="weight",
                resolution=resolution,
                beta=0.01,
                initial_membership=None,
                n_iterations=2,
            )
            membership = np.asarray(partition.membership)
            communities = np.stack([
                membership == membership[channel]
                for channel in range(channels)
            ])
            bank.append((communities, resolution))
        return tuple(bank)

    def _partition_fingerprint(self, spatial_distances):
        settings = {
            "graph_neighbors": self.graph_neighbors,
            "resolution_values": self.resolution_values,
            "distance_scale_degrees": self.distance_scale_degrees,
            "partition_seed": self.partition_seed,
            "channels": int(spatial_distances.shape[0]),
        }
        digest = hashlib.sha256(
            json.dumps(settings, sort_keys=True).encode("utf-8")
            + np.asarray(spatial_distances, dtype=np.float64).tobytes()
        ).hexdigest()
        return digest

    def partition_cache_path(self, spatial_distances):
        cache_root = Path(__file__).resolve().parents[2] / "artifacts" / "masking"
        return cache_root / f"leiden_partitions_{self._partition_fingerprint(spatial_distances)}.npz"

    def precompute_partition_cache(self, spatial_distances):
        """Compute and persist the deterministic CPU-only Leiden bank."""
        try:
            import igraph as ig
        except ImportError as error:
            raise RuntimeError(
                "leiden_multiscale masking requires python-igraph"
            ) from error
        ig.set_random_number_generator(random.Random(self.partition_seed))
        bank = self._partition_bank(spatial_distances)
        destination = self.partition_cache_path(spatial_distances)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            fingerprint=self._partition_fingerprint(spatial_distances),
            communities=np.stack([item[0] for item in bank]),
            resolutions=np.asarray([item[1] for item in bank], dtype=np.float64),
        )
        temporary.replace(destination)
        return destination

    def _load_partition_cache(self, spatial_distances):
        path = self.partition_cache_path(spatial_distances)
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as payload:
            expected = self._partition_fingerprint(spatial_distances)
            observed = str(payload["fingerprint"].item())
            communities = payload["communities"]
            resolutions = payload["resolutions"]
        if observed != expected:
            raise RuntimeError(f"Leiden partition cache fingerprint mismatch: {path}")
        if communities.shape != (
            len(self.resolution_values), spatial_distances.shape[0],
            spatial_distances.shape[0],
        ):
            raise RuntimeError(f"invalid Leiden partition cache shape: {path}")
        if not np.array_equal(resolutions, np.asarray(self.resolution_values)):
            raise RuntimeError(f"Leiden partition cache resolution mismatch: {path}")
        return tuple(
            (communities[index].astype(np.bool_, copy=False), float(resolution))
            for index, resolution in enumerate(resolutions)
        )

    def _scale_options(self, partition_bank, channels):
        options_by_scale = []
        for minimum, maximum in self.time_ranges:
            options = []
            for communities, resolution in partition_bank:
                for seed_channel in range(channels):
                    selected_indices = np.flatnonzero(
                        communities[seed_channel]
                    )
                    for duration in range(minimum, maximum + 1):
                        options.append((
                            selected_indices,
                            duration,
                            int(selected_indices.size * duration),
                            resolution,
                        ))
            options_by_scale.append(options)
        return tuple(options_by_scale)

    def _sample_one(self, options_by_scale, channels, patches, rng):
        per_scale_target = round(channels * patches * self.mask_ratio / 3.0)
        maximum_overshoot = (
            math.floor(channels * patches * self.max_mask_ratio)
            - 3 * per_scale_target
        )

        for restart in range(self.max_sample_restarts):
            target = np.zeros((channels, patches), dtype=np.bool_)
            scale_counts = [None] * len(self.time_ranges)
            block_counts = [None] * len(self.time_ranges)
            duration_sums = [None] * len(self.time_ranges)
            community_sums = [None] * len(self.time_ranges)
            resolution_sums = [None] * len(self.time_ranges)
            overlap_rejections = 0
            overshoot_rejections = 0
            used_overshoot = 0
            failed = False
            # Place the longest intact blocks first to avoid fragmenting the
            # grid; reported scale order remains small, medium, large.
            for scale_index in reversed(range(len(self.time_ranges))):
                minimum, maximum = self.time_ranges[scale_index]
                added = 0
                blocks = 0
                duration_sum = 0
                community_sum = 0
                resolution_sum = 0.0
                attempts = 0
                while added < per_scale_target:
                    contribution_limit = (
                        per_scale_target - added
                        + maximum_overshoot - used_overshoot
                    )
                    eligible = [
                        option for option in options_by_scale[scale_index]
                        if option[2] <= contribution_limit
                    ]
                    if not eligible:
                        failed = True
                        break
                    chosen = None
                    for option_index in rng.permutation(len(eligible)):
                        attempts += 1
                        if attempts > self.max_block_attempts:
                            break
                        selected_indices, duration, contribution, resolution = (
                            eligible[int(option_index)]
                        )
                        occupied_time = target[selected_indices].any(axis=0)
                        valid_starts = [
                            start
                            for start in range(patches - duration + 1)
                            if not occupied_time[start:start + duration].any()
                        ]
                        if valid_starts:
                            start = valid_starts[
                                int(rng.integers(0, len(valid_starts)))
                            ]
                            chosen = (
                                selected_indices,
                                duration,
                                contribution,
                                resolution,
                                start,
                            )
                            break
                        overlap_rejections += 1
                    if chosen is None:
                        failed = True
                        break
                    selected_indices, duration, contribution, resolution, start = chosen
                    stop = start + duration
                    target[selected_indices[:, None], np.arange(start, stop)] = True
                    added += contribution
                    blocks += 1
                    duration_sum += duration
                    community_sum += int(selected_indices.size)
                    resolution_sum += resolution
                if failed:
                    break
                scale_counts[scale_index] = added
                block_counts[scale_index] = blocks
                duration_sums[scale_index] = duration_sum
                community_sums[scale_index] = community_sum
                resolution_sums[scale_index] = resolution_sum
                used_overshoot += added - per_scale_target
            if not failed:
                if int(target.sum()) != sum(scale_counts):
                    raise RuntimeError("Leiden community blocks overlapped")
                return {
                    "mask": target,
                    "scale_counts": scale_counts,
                    "block_counts": block_counts,
                    "duration_sums": duration_sums,
                    "community_sums": community_sums,
                    "resolution_sums": resolution_sums,
                    "overlap_rejections": overlap_rejections,
                    "overshoot_rejections": overshoot_rejections,
                    "restarts": restart,
                }
        raise RuntimeError(
            "Leiden multiscale masking failed to place non-overlapping blocks"
        )

    def __call__(
        self,
        batch_size,
        num_latents,
        num_patches,
        valid_token_mask,
        channel_coordinates=None,
        generator=None,
        **_,
    ):
        batch, channels, patches = _validate_fixed_grid(
            batch_size, num_latents, num_patches, valid_token_mask
        )
        coordinates = GeometryTubeletMaskingPolicy._coordinates(
            channel_coordinates, batch, channels
        )
        coordinate_array = coordinates.numpy()
        cosine = np.matmul(coordinate_array, coordinate_array.transpose(0, 2, 1))
        spatial_distances = np.arccos(np.clip(cosine, -1.0, 1.0))
        numpy_seed = int(torch.randint(
            0,
            torch.iinfo(torch.int64).max,
            (),
            generator=generator,
            dtype=torch.int64,
        ).item())
        rng = np.random.default_rng(numpy_seed)
        try:
            import igraph as ig
        except ImportError as error:
            raise RuntimeError(
                "leiden_multiscale masking requires python-igraph"
            ) from error
        # The community bank is a deterministic part of the masking policy;
        # only block selection and placement consume the epoch/rank RNG.
        ig.set_random_number_generator(random.Random(self.partition_seed))

        distance_key = spatial_distances[0].tobytes()
        if distance_key != self._cached_distance_key:
            precomputed = self._load_partition_cache(spatial_distances[0])
            self._partition_cache_hit = precomputed is not None
            self._cached_partition_bank = (
                precomputed
                if precomputed is not None
                else self._partition_bank(spatial_distances[0])
            )
            self._cached_scale_options = self._scale_options(
                self._cached_partition_bank, channels
            )
            self._cached_distance_key = distance_key
        first_bank = self._cached_partition_bank
        if np.allclose(spatial_distances, spatial_distances[:1]):
            banks = [first_bank] * batch
        else:
            banks = [first_bank] + [
                self._partition_bank(spatial_distances[index])
                for index in range(1, batch)
            ]
        option_sets = {}
        samples = []
        for index in range(batch):
            bank_identity = id(banks[index])
            if bank_identity not in option_sets:
                option_sets[bank_identity] = (
                    self._cached_scale_options
                    if banks[index] is first_bank
                    else self._scale_options(banks[index], channels)
                )
            samples.append(self._sample_one(
                option_sets[bank_identity], channels, patches, rng
            ))
        target_mask = torch.from_numpy(np.stack([
            sample["mask"] for sample in samples
        ])).to(valid_token_mask.device)
        context_mask = valid_token_mask & ~target_mask
        names = ("small", "medium", "large")
        diagnostics = {
            "leiden_total_target_tokens": target_mask.sum((1, 2)).float().mean(),
            "leiden_total_mask_ratio": target_mask.float().mean(),
            "leiden_overlap_rejections": torch.tensor([
                sample["overlap_rejections"] for sample in samples
            ]).float().mean(),
            "leiden_overshoot_rejections": torch.tensor([
                sample["overshoot_rejections"] for sample in samples
            ]).float().mean(),
            "leiden_sample_restarts": torch.tensor([
                sample["restarts"] for sample in samples
            ]).float().mean(),
            "leiden_partition_cache_hit": torch.tensor(
                float(self._partition_cache_hit)
            ),
        }
        for scale_index, name in enumerate(names):
            scale_tokens = torch.tensor([
                sample["scale_counts"][scale_index] for sample in samples
            ]).float()
            blocks = torch.tensor([
                sample["block_counts"][scale_index] for sample in samples
            ]).float()
            duration_sum = torch.tensor([
                sample["duration_sums"][scale_index] for sample in samples
            ]).float()
            community_sum = torch.tensor([
                sample["community_sums"][scale_index] for sample in samples
            ]).float()
            resolution_sum = torch.tensor([
                sample["resolution_sums"][scale_index] for sample in samples
            ]).float()
            diagnostics[f"leiden_{name}_tokens"] = scale_tokens.mean()
            diagnostics[f"leiden_{name}_blocks"] = blocks.mean()
            diagnostics[f"leiden_{name}_mean_duration"] = (
                duration_sum / blocks
            ).mean()
            diagnostics[f"leiden_{name}_mean_community_size"] = (
                community_sum / blocks
            ).mean()
            diagnostics[f"leiden_{name}_mean_resolution"] = (
                resolution_sum / blocks
            ).mean()
        return _mask_result(
            context_mask,
            target_mask[:, None],
            valid_token_mask,
            masking_diagnostics=diagnostics,
        )


def build_masking_policy(config):
    """Build the configured EEG pretraining masking policy."""
    policy = config["policy"]
    if policy == "geometry_tubelet":
        return GeometryTubeletMaskingPolicy(
            mask_ratio=config["mask_ratio"],
            min_radius_degrees=config["min_radius_degrees"],
            max_radius_degrees=config["max_radius_degrees"],
            min_time_patches=config["min_time_patches"],
            max_time_patches=config["max_time_patches"],
        )
    if policy == "leiden_multiscale":
        return LeidenMultiScaleMaskingPolicy(
            mask_ratio=config["mask_ratio"],
            max_mask_ratio=config["max_mask_ratio"],
            time_ranges=config["time_ranges"],
            graph_neighbors=config["graph_neighbors"],
            resolution_values=config["resolution_values"],
            distance_scale_degrees=config["distance_scale_degrees"],
            partition_seed=config["partition_seed"],
            max_block_attempts=config["max_block_attempts"],
            max_sample_restarts=config["max_sample_restarts"],
            allow_overlap=config["allow_overlap"],
        )
    if policy == "random_patch":
        return RandomPatchMaskingPolicy(mask_ratio=config["mask_ratio"])
    if policy == "ijepa_multiblock":
        return IJEPAMultiBlockMaskingPolicy(
            context_scale=config["context_scale"],
            target_scale=config["target_scale"],
            target_aspect_ratio=config["target_aspect_ratio"],
            num_context_blocks=config["num_context_blocks"],
            num_target_blocks=config["num_target_blocks"],
            min_context_tokens=config["min_context_tokens"],
            allow_overlap=config["allow_overlap"],
        )
    raise ValueError(f"unsupported masking policy: {policy}")
