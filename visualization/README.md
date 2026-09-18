# SHPE on the fsaverage head

This directory renders the model's 25 real spherical-harmonic basis functions
(degrees 0–4) on MNE's bundled `fsaverage` scalp surface. Camera position is
fixed at azimuth 30° and elevation 25°. The dots mark the 30 scalp channels
used by `FACEDDataset`; channel names are omitted.

## Regenerate

From `E:\workspace\visualization`:

```powershell
E:\workspace\.venv-ablation\Scripts\python.exe prepare_fsaverage_mesh.py
E:\workspace\.venv-ablation\Scripts\python.exe render_shpe_head.py
```

The scripts require NumPy, Matplotlib, and MNE-Python. `fsaverage_head_mesh.json`
is bundled so rendering only needs NumPy and Matplotlib; rerun
`prepare_fsaverage_mesh.py` when changing the montage or MNE version.

## Files

- `output/faced_electrodes_az30_el25.png`: neutral head with FACED electrodes.
- `output/shpe_00_l0_m+0_az30_el25.png` through
  `output/shpe_24_l4_m+4_az30_el25.png`: one image per SH basis. Filenames
  report the zero-based basis index, degree, order, azimuth, and elevation.
- `output/legend_horizontal_RdBu_r_minus1_to_plus1.png` and
  `output/legend_vertical_RdBu_r_minus1_to_plus1.png`: horizontal and vertical
  versions of the shared color legend.
- `output/legend_ranges.csv` and `.json`: actual minimum/maximum on the scalp
  mesh, actual minimum/maximum at the 30 FACED electrodes, and fixed colorbar
  limits (`legend_vmin=-1`, `legend_vmax=1`) for every image.

The colorbar is **−1 to +1 for every basis**, so the same color has the same
SH value across all 25 images. The plots contain no title or legend and are
cropped to the head with a 3 mm margin. Actual extrema are recorded separately.

## Coordinate mapping

MNE's `standard_1020` electrode positions originate in the montage MRI frame.
`prepare_fsaverage_mesh.py` applies the montage's fiducial-based native-to-head
transform before snapping each electrode to the nearest `fsaverage` scalp
vertex. The head surface is transformed from fsaverage MRI to head coordinates
for display. SH values are evaluated in the original MRI/montage coordinates,
matching `src/modules/position_embedding.py`, which normalizes those raw
coordinates before evaluating the real SH basis.

The surface is a template head and the electrode positions are standard montage
positions, not subject-specific digitizations. The surface colors illustrate
the continuous SH basis; the model evaluates it only at its electrode positions.
