"""GR6 geometry tubelet 알고리즘. 원본의 sampling·합집합·마지막 영역 절단을 보존한다."""

import math
import mne
import numpy as np
import torch
import torch.nn.functional as F


def physical_channel_coordinates(channel_names):
    """MNE standard_1020의 실제 미터 좌표. SHPE용 정규화 좌표와 구분한다."""
    from src.data.electrode_geometry import canonicalize_channel_name
    positions = mne.channels.make_standard_montage("standard_1020").get_positions()["ch_pos"]
    positions = {name.upper(): position for name, position in positions.items()}
    names = [canonicalize_channel_name(name) for name in channel_names]
    missing = [name for name in names if name not in positions]
    if missing:
        raise ValueError("missing physical masking coordinates: " + str(missing))
    return torch.tensor(np.stack([positions[name] for name in names]), dtype=torch.float32)


class GeometryTubeletMaskingPolicy:
    """Mask unions of local electrode circles and contiguous time intervals.

    A proposal selects a random electrode as its spatial center and combines
    its neighborhood with a contiguous patch interval. Historical geometry uses
    geodesic angles; the REVE radius option uses physical Euclidean metres. Proposals are added
    until the exact requested target count is reached. If the final proposal
    crosses the count boundary, its most central space-time tokens are retained.
    """

    def __init__(
        self,
        mask_ratio,
        min_radius_degrees=None,
        max_radius_degrees=None,
        min_time_patches=2,
        max_time_patches=15,
        distance_metric="geodesic",
        radius_m=None,
    ):
        self.mask_ratio = float(mask_ratio)
        self.distance_metric = distance_metric
        if distance_metric == "euclidean_m":
            if radius_m is None or not math.isfinite(radius_m) or radius_m <= 0:
                raise ValueError("euclidean masking requires a positive radius_m")
            self.min_radius = self.max_radius = float(radius_m)
        elif distance_metric == "geodesic":
            self.min_radius = float(min_radius_degrees)
            self.max_radius = float(max_radius_degrees)
            if not 0.0 < self.min_radius <= self.max_radius < 180.0:
                raise ValueError("geometry radii must satisfy 0 < min <= max < 180")
        else:
            raise ValueError("unsupported geometry distance_metric: " + str(distance_metric))
        self.min_time_patches = int(min_time_patches)
        self.max_time_patches = int(max_time_patches)
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be between zero and one")
        if not 1 <= self.min_time_patches <= self.max_time_patches:
            raise ValueError("time spans must satisfy 1 <= min <= max")

    @staticmethod
    def _coordinates(channel_coordinates, batch, channels, normalize=True):
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
            if normalize:
                coordinates = F.normalize(coordinates, dim=-1)
            return coordinates.unsqueeze(0).expand(batch, -1, -1)
        if coordinates.shape != (batch, channels, 3):
            raise ValueError("channel_coordinates must have shape [C,3] or [B,C,3]")
        return F.normalize(coordinates, dim=-1) if normalize else coordinates

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
            radius_value = float(rng.uniform(
                self.min_radius, self.max_radius
            ))
            radius = math.radians(radius_value) if self.distance_metric == "geodesic" else radius_value
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
            radius_sum += radius_value
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
        batch, channels, patches = int(batch_size), int(num_latents), int(num_patches)
        if valid_token_mask.dtype != torch.bool or valid_token_mask.shape != (batch, channels, patches):
            raise ValueError("geometry masking requires a boolean [B,C,T] valid mask")
        if not valid_token_mask.all():
            raise ValueError("geometry masking requires the fixed valid pretraining grid")
        if self.min_time_patches > patches:
            raise ValueError("minimum tubelet time span exceeds the patch grid")
        coordinates = GeometryTubeletMaskingPolicy._coordinates(
            channel_coordinates, batch, channels, normalize=self.distance_metric == "geodesic"
        )
        coordinate_array = coordinates.numpy()
        if self.distance_metric == "euclidean_m":
            # REVE의 KDTree.query_ball_point(..., 0.03)와 같은 3D 유클리드 이웃.
            delta = coordinate_array[:, :, None, :] - coordinate_array[:, None, :, :]
            spatial_distances = np.linalg.norm(delta, axis=-1)
        else:
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
            ("geometry_mean_radius_m" if self.distance_metric == "euclidean_m"
             else "geometry_mean_radius_degrees"): torch.tensor(mean_radii).mean(),
            "geometry_mean_time_span_patches": torch.tensor(mean_durations).mean(),
            "geometry_clipped_tubelet_count": (
                torch.tensor(clipped_counts).float().mean()
            ),
        }
        # 모든 tubelet의 합집합을 하나의 target block으로 복원한다.
        # 겹친 패치는 한 번만 포함하고 나머지 모든 패치를 context로 사용한다.
        return {
            "context_mask": context_mask,
            "target_mask": target_mask,
            "target_blocks": target_mask[:, None],
            "target_token_valid": torch.ones(batch, 1, target_count, dtype=torch.bool,
                                             device=valid_token_mask.device),
            "masking_diagnostics": diagnostics,
        }

