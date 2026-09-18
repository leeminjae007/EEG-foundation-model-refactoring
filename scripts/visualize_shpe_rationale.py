"""CPU-only geometry diagnostics; no EEG data or performance claims."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy.special import eval_legendre
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.modules.position_embedding import real_spherical_harmonic_features, PositionEmbedding
from ablation.sources import reve_position_source


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def sh(x, degree=4):
    return real_spherical_harmonic_features(torch.as_tensor(x, dtype=torch.float32), degree).numpy()


def cos_sim(a, b):
    return (unit(a) * unit(b)).sum(-1)


def save(fig, folder, stem):
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(folder / (stem + '.' + ext), dpi=200, facecolor='white')
    plt.close(fig)


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    rng = np.random.default_rng(20260917)
    grid = np.linspace(-1, 1, 241)
    xx, yy = np.meshgrid(grid, grid)
    valid = xx * xx + yy * yy <= 1
    zz = np.sqrt(np.clip(1 - xx * xx - yy * yy, 0, None))
    xyz = np.stack([xx[valid], yy[valid], zz[valid]], -1)
    basis = sh(xyz)
    fig, axes = plt.subplots(1, 5, figsize=(13, 4.1))
    fig.subplots_adjust(left=.035, right=.91, bottom=.24, top=.76, wspace=.24)
    fig.text(.035, .94, 'SHPE exposes spatial modes at several scales', fontsize=19, weight='bold')
    fig.text(.035, .86, 'Actual SH basis in the repository; representative modes on an idealized upper hemisphere', color='#4b5563')
    for degree, ax in enumerate(axes):
        order = degree
        column = degree * degree + (order + degree)
        values = basis[:, column]
        image = np.full(xx.shape, np.nan)
        image[valid] = values / np.max(np.abs(values))
        artist = ax.imshow(image, origin='lower', extent=(-1, 1, -1, 1), cmap='RdBu_r', vmin=-1, vmax=1)
        ax.add_patch(Circle((0, 0), 1, fill=False, lw=.7, color='#374151'))
        ax.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05))
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(r'$\ell=%d$' % degree, fontsize=15)
        ax.text(.5, -.09, '%d %s at this degree' % (2 * degree + 1, 'mode' if degree == 0 else 'modes'), ha='center', transform=ax.transAxes)
        ax.text(.5, -.22, '%d cumulative' % ((degree + 1) ** 2), ha='center', color='#4b5563', transform=ax.transAxes)
    cax = fig.add_axes([.945, .28, .014, .43])
    bar = fig.colorbar(artist, cax=cax, ticks=[-1, 0, 1])
    bar.set_label('Per-map normalized amplitude')
    fig.text(.035, .10, 'Affine XYZ on a unit sphere spans degrees 0–1. SH through degree 4 adds 21 higher-order modes.', fontsize=11)
    fig.text(.035, .035, 'Basis illustration only: deterministic features add no new coordinate information; this is not a comparison with REVE Fourier PE.', fontsize=10, color='#4b5563')
    save(fig, output, 'shpe-spatial-modes')

    # Equal angular separation, different global orientations. No training or
    # method-dependent tuning: 1,024 pairs per angle, one predetermined radius.
    pairs = 1024
    anchor = unit(rng.normal(size=(pairs, 3)))
    tangent = rng.normal(size=(pairs, 3))
    tangent = unit(tangent - (tangent * anchor).sum(-1, keepdims=True) * anchor)
    angles = np.linspace(0, 180, 91)
    radius = .09
    module = reve_position_source()
    fourier = {dim: module.FourierEmb4D(dim, freqs=4) for dim in (200, 512)}
    def fv(x, dim):
        coords = np.concatenate([radius * x, np.zeros((len(x), 1))], -1)
        return fourier[dim](torch.tensor(coords[None], dtype=torch.float32))[0].numpy()
    first_sh = sh(anchor)
    first_fourier = {dim: fv(anchor, dim) for dim in fourier}
    similarities = {'sh25': [], 'fourier200': [], 'fourier512': []}
    for angle in angles:
        radians = np.deg2rad(angle)
        other = np.cos(radians) * anchor + np.sin(radians) * tangent
        similarities['sh25'].append(cos_sim(first_sh, sh(other)))
        for dim in fourier:
            similarities['fourier' + str(dim)].append(cos_sim(first_fourier[dim], fv(other, dim)))
    similarities = {name: np.asarray(values) for name, values in similarities.items()}
    exact = sum((2 * degree + 1) * eval_legendre(degree, np.cos(np.deg2rad(angles)))
                for degree in range(5)) / 25
    sh_error = float(np.max(np.abs(similarities['sh25'] - exact[:, None])))
    assert sh_error < 2e-5, sh_error

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    fig.subplots_adjust(left=.07, right=.975, top=.70, bottom=.26, wspace=.27)
    fig.text(.07, .94, 'A geometry prior that can be tested directly', fontsize=19, weight='bold')
    fig.text(.07, .875, 'Fixed bases only — before learned projections, GELU, or normalization', fontsize=12, color='#374151')
    colors = {'sh25': '#147d92', 'fourier200': '#ca6924', 'fourier512': '#6950a1'}
    labels = {'sh25': 'SH, degrees 0–4 (25 features)',
              'fourier200': 'REVE Fourier (200 features; adapter)',
              'fourier512': 'REVE Fourier (512; complete grid)'}
    ax = axes[0]
    for name, values in similarities.items():
        mean = values.mean(1)
        lo, hi = np.percentile(values, [5, 95], axis=1)
        ax.plot(angles, mean, color=colors[name], lw=2, label=labels[name])
        ax.fill_between(angles, lo, hi, color=colors[name], alpha=.15, linewidth=0)
    ax.set(xlabel='Angular separation (degrees)', ylabel='Cosine similarity (unitless)',
           xlim=(0, 180), ylim=(-.23, 1.06), xticks=[0, 45, 90, 135, 180])
    ax.set_title('A  Similarity versus separation', loc='left', fontsize=13, pad=12)
    ax.grid(alpha=.17)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, fontsize=9.4,
               loc='upper center', bbox_to_anchor=(.53, .835), ncol=3)
    ax = axes[1]
    for name, values in similarities.items():
        ax.plot(angles, values.std(1), color=colors[name], lw=2, label=labels[name])
    ax.set(xlabel='Angular separation (degrees)', ylabel='SD across orientations (unitless)',
           xlim=(0, 180), xticks=[0, 45, 90, 135, 180])
    ax.set_ylim(bottom=-.008)
    ax.set_title('B  Dependence on global orientation', loc='left', fontsize=13, pad=12)
    ax.grid(alpha=.17)
    fig.text(.07, .13, '1,024 random orientations per angle; sphere radius 9 cm; equal time t=0; shading: 5th–95th percentile.', fontsize=10.5)
    fig.text(.07, .075, 'The SH inner product depends only on angle. Its sidelobes mean that similarity is not monotonic with distance.', fontsize=10.5)
    fig.text(.07, .025, 'This diagnoses the fixed basis. It does not establish that the final learned SHPE is rotation-invariant or more accurate.', fontsize=10.5, color='#4b5563')
    save(fig, output, 'shpe-geometry-kernel')

    # Confirm the raw-coordinate claim by fitting degree-1 SH from [1,x,y,z]
    # and evaluate on a separate uniform point set. The figures do not contain
    # a synthetic task chosen to maximize a gap between methods.
    fit_x = unit(rng.normal(size=(1024, 3)))
    eval_x = unit(rng.normal(size=(1024, 3)))
    coefficients = np.linalg.lstsq(np.c_[np.ones(len(fit_x)), fit_x], sh(fit_x, 1), rcond=None)[0]
    degree1_error = float(np.max(np.abs(np.c_[np.ones(len(eval_x)), eval_x] @ coefficients - sh(eval_x, 1))))
    assert degree1_error < 2e-6
    radial_error = float(np.max(np.abs(sh(anchor) - sh(anchor * 1.2))))
    assert radial_error < 2e-5
    # Learned full-PE behavior is deliberately not equated to the basis theorem.
    torch.manual_seed(20260917)
    pe = PositionEmbedding(100, 100, 10000., 1e-6).eval()
    q = np.cos(np.pi/3)*anchor + np.sin(np.pi/3)*tangent
    with torch.no_grad():
        full_a = pe(torch.tensor(anchor, dtype=torch.float32), 1, 1, torch.float32)[0, :, 0].numpy()
        full_b = pe(torch.tensor(q, dtype=torch.float32), 1, 1, torch.float32)[0, :, 0].numpy()
    proof = dict(seed=20260917, experiment_type='CPU-only synthetic geometry diagnostic; no EEG performance evaluation',
                 max_degree=4, sh_feature_count=25, sphere_radius_m=radius, pairs_per_angle=pairs,
                 angles_degrees=angles.tolist(), exact_addition_theorem_max_abs_error=sh_error,
                 xyz_affine_spans_degree01_max_abs_error=degree1_error,
                 radial_scaling_1_2_basis_max_abs_error=radial_error,
                 random_initialized_full_shpe_similarity_sd_at_60_degrees=float(cos_sim(full_a, full_b).std()),
                 fixed_basis_sd_at_60_degrees={k:float(v[30].std()) for k,v in similarities.items()},
                 excluded_components='REVE learned MLP and LayerNorm; SH learned projection, GELU and RMSNorm; temporal branch',
                 source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [
                     'src/modules/position_embedding.py','ablation/vendor/reve/src/models/encoder.py']})
    (output / 'diagnostics.json').write_text(json.dumps(proof, indent=2)+'\n')
    np.savez_compressed(output / 'geometry_samples.npz', angles_degrees=angles, **similarities)
    print(json.dumps({k:v for k,v in proof.items() if k!='angles_degrees'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/figures/shpe-rationale-20260917')
    main(parser.parse_args().output)
