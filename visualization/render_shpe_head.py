"""Render the 25 real SH basis functions on the fsaverage scalp.

Run: python render_shpe_head.py
Requires: numpy, matplotlib. The bundled fsaverage_head_mesh.json is required.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.collections import PolyCollection


ROOT = Path(__file__).resolve().parent
MESH = ROOT / "fsaverage_head_mesh.json"
OUTPUT = ROOT / "output"
AZIMUTH_DEG = 30.0
ELEVATION_DEG = 25.0
ELECTRODE_DOT_SIZE = 150
IMAGE_DPI = 180
IMAGE_INCHES = 9
CENTER = np.array([0.0, -0.005, 0.005])
HEAD_MARGIN_M = 0.003
COLOR_VMIN = -1.0
COLOR_VMAX = 1.0
ANGLE_TAG = f"az{AZIMUTH_DEG:02.0f}_el{ELEVATION_DEG:02.0f}"


def associated_legendre(degree: int, order: int, z: np.ndarray) -> np.ndarray:
    """Match src/modules/position_embedding.py's Condon-Shortley basis."""
    value = np.ones_like(z)
    if order:
        sine_theta = np.sqrt(np.maximum(0.0, (1.0 - z) * (1.0 + z)))
        factor = 1.0
        for _ in range(order):
            value = -factor * sine_theta * value
            factor += 2.0
    if degree == order:
        return value
    current = (2 * order + 1) * z * value
    if degree == order + 1:
        return current
    previous = value
    for current_degree in range(order + 2, degree + 1):
        following = (
            (2 * current_degree - 1) * z * current
            - (current_degree + order - 1) * previous
        ) / (current_degree - order)
        previous, current = current, following
    return current


def spherical_harmonics(points: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    xyz = points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-8)
    x, y, z = xyz.T
    phi = np.arctan2(y, x)
    labels = []
    features = []
    for degree in range(5):
        for order in range(-degree, degree + 1):
            absolute_order = abs(order)
            polynomial = associated_legendre(degree, absolute_order, z)
            scale = math.sqrt(
                (2 * degree + 1) / (4.0 * math.pi)
                * math.factorial(degree - absolute_order)
                / math.factorial(degree + absolute_order)
            )
            if order < 0:
                basis = math.sqrt(2.0) * scale * polynomial * np.sin(absolute_order * phi)
            elif order == 0:
                basis = scale * polynomial
            else:
                basis = math.sqrt(2.0) * scale * polynomial * np.cos(order * phi)
            labels.append((degree, order))
            features.append(basis)
    return np.stack(features, axis=1), labels


def camera_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw = math.radians(AZIMUTH_DEG)
    elev = math.radians(ELEVATION_DEG)
    right = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
    up = np.array([-math.sin(elev) * math.sin(yaw),
                   -math.sin(elev) * math.cos(yaw), math.cos(elev)])
    forward = np.array([math.cos(elev) * math.sin(yaw),
                        math.cos(elev) * math.cos(yaw), math.sin(elev)])
    return right, up, forward


def project(points: np.ndarray) -> np.ndarray:
    right, up, forward = camera_basis()
    shifted = points - CENTER
    return np.column_stack((shifted @ right, shifted @ up, shifted @ forward))


def face_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a, b, c = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
    normal = np.cross(b - a, c - a)
    return normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-10)


def visible_electrodes(projected_vertices: np.ndarray,
                       triangles: np.ndarray,
                       projected_electrodes: np.ndarray) -> np.ndarray:
    """Orthographic mesh depth test; do not draw dots through the head."""
    tri = projected_vertices[triangles]
    a, b, c = tri[:, 0, :2], tri[:, 1, :2], tri[:, 2, :2]
    denominator = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    safe = np.abs(denominator) > 1e-12
    shown = []
    for x, y, depth in projected_electrodes:
        w1 = np.zeros(len(tri))
        w2 = np.zeros(len(tri))
        w1[safe] = ((b[safe, 1] - c[safe, 1]) * (x - c[safe, 0])
                    + (c[safe, 0] - b[safe, 0]) * (y - c[safe, 1])) / denominator[safe]
        w2[safe] = ((c[safe, 1] - a[safe, 1]) * (x - c[safe, 0])
                    + (a[safe, 0] - c[safe, 0]) * (y - c[safe, 1])) / denominator[safe]
        w3 = 1 - w1 - w2
        inside = safe & (w1 >= -1e-6) & (w2 >= -1e-6) & (w3 >= -1e-6)
        if not np.any(inside):
            shown.append(depth > 0)
            continue
        surface_depth = (w1 * tri[:, 0, 2] + w2 * tri[:, 1, 2]
                         + w3 * tri[:, 2, 2])[inside].max()
        shown.append(depth >= surface_depth - 0.003)
    return np.asarray(shown, dtype=bool)


def render(
    vertices: np.ndarray,
    triangles: np.ndarray,
    electrode_rows: list[dict],
    values: np.ndarray | None,
    output: Path,
) -> None:
    projected = project(vertices)
    polygons = projected[triangles, :2]
    depth = projected[triangles, 2].mean(axis=1)
    order = np.argsort(depth)  # far triangles first
    normal = face_normals(vertices, triangles)
    _, _, forward = camera_basis()
    light = 0.72 + 0.28 * np.abs(normal @ forward)
    if values is None:
        rgb = np.broadcast_to(np.array([0.94, 0.77, 0.63]), (len(triangles), 3)).copy()
    else:
        face_value = values[triangles].mean(axis=1)
        rgb = plt.get_cmap("RdBu_r")(colors.Normalize(COLOR_VMIN, COLOR_VMAX)(face_value))[:, :3]
    rgb = rgb * light[:, None]

    x_min = float(projected[:, 0].min() - HEAD_MARGIN_M)
    x_max = float(projected[:, 0].max() + HEAD_MARGIN_M)
    y_min = float(projected[:, 1].min() - HEAD_MARGIN_M)
    y_max = float(projected[:, 1].max() + HEAD_MARGIN_M)
    width, height = x_max - x_min, y_max - y_min
    fig = plt.figure(figsize=(IMAGE_INCHES * width / height, IMAGE_INCHES),
                     dpi=IMAGE_DPI, facecolor="white")
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.add_collection(PolyCollection(polygons[order], facecolors=rgb[order], edgecolors="none", antialiased=False))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")

    electrode_points = np.array([row["position_head"] for row in electrode_rows], dtype=float)
    projected_electrodes = project(electrode_points)
    visible = visible_electrodes(projected, triangles, projected_electrodes)
    ax.scatter(projected_electrodes[visible, 0], projected_electrodes[visible, 1],
               s=ELECTRODE_DOT_SIZE, facecolors="#153c67", edgecolors="white",
               linewidths=2.0, zorder=10)

    fig.savefig(output, dpi=IMAGE_DPI, facecolor="white")
    plt.close(fig)


def render_legend(output: Path, orientation: str) -> None:
    if orientation == "horizontal":
        fig = plt.figure(figsize=(7, 1.1), dpi=IMAGE_DPI, facecolor="white")
        ax = fig.add_axes((0.06, 0.45, 0.88, 0.28))
    elif orientation == "vertical":
        fig = plt.figure(figsize=(1.4, 7), dpi=IMAGE_DPI, facecolor="white")
        ax = fig.add_axes((0.22, 0.06, 0.28, 0.88))
    else:
        raise ValueError(f"Unsupported legend orientation: {orientation}")
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=colors.Normalize(COLOR_VMIN, COLOR_VMAX), cmap="RdBu_r"),
        cax=ax, orientation=orientation,
    )
    colorbar.set_ticks([COLOR_VMIN, 0.0, COLOR_VMAX])
    colorbar.ax.tick_params(labelsize=14)
    colorbar.set_label("Real spherical harmonic value", fontsize=13)
    fig.savefig(output, dpi=IMAGE_DPI, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    mesh = json.loads(MESH.read_text(encoding="utf-8"))
    vertices = np.asarray(mesh["vertices"], dtype=float)
    sh_coordinates = np.asarray(mesh["sh_coordinates"], dtype=float)
    triangles = np.asarray(mesh["triangles"], dtype=np.int32)
    electrodes = mesh["electrodes"]
    electrode_xyz = np.asarray([row["source_position_mri"] for row in electrodes], dtype=float)
    surface_values, labels = spherical_harmonics(sh_coordinates)
    electrode_values, electrode_labels = spherical_harmonics(electrode_xyz)
    assert labels == electrode_labels and surface_values.shape[1] == 25
    OUTPUT.mkdir(parents=True, exist_ok=True)

    render(vertices, triangles, electrodes, None,
           OUTPUT / f"faced_electrodes_{ANGLE_TAG}.png")
    legend_names = {
        "horizontal": "legend_horizontal_RdBu_r_minus1_to_plus1.png",
        "vertical": "legend_vertical_RdBu_r_minus1_to_plus1.png",
    }
    for orientation, name in legend_names.items():
        render_legend(OUTPUT / name, orientation)
    rows = []
    for index, (degree, order) in enumerate(labels):
        vals = surface_values[:, index]
        electrode_vals = electrode_values[:, index]
        filename = f"shpe_{index:02d}_l{degree}_m{order:+d}_{ANGLE_TAG}.png"
        render(vertices, triangles, electrodes, vals, OUTPUT / filename)
        rows.append({
            "basis_index": index,
            "degree_l": degree,
            "order_m": order,
            "image": filename,
            "surface_min": float(vals.min()),
            "surface_max": float(vals.max()),
            "electrode_min": float(electrode_vals.min()),
            "electrode_max": float(electrode_vals.max()),
            "legend_vmin": COLOR_VMIN,
            "legend_vmax": COLOR_VMAX,
        })
        print(filename)
    with (OUTPUT / "legend_ranges.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "legend_ranges.json").write_text(
        json.dumps({"azimuth_deg": AZIMUTH_DEG, "elevation_deg": ELEVATION_DEG,
                    "colormap": "RdBu_r", "legend_images": legend_names,
                    "ranges": rows}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
