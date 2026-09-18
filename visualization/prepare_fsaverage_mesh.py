"""Bundle MNE's fsaverage scalp and correctly aligned standard_1020 electrodes.

Run from a Python environment with MNE, NumPy installed:
    python prepare_fsaverage_mesh.py
"""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np


ROOT = Path(__file__).resolve().parent
# Exact 30-channel acquisition order used by FACEDDataset. A2/A1 mastoid
# references are excluded from the model's 30 scalp channels.
CHANNELS = (
    "FP1", "FP2", "FZ", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5",
    "FC6", "CZ", "C3", "C4", "T7", "T8", "CP1", "CP2", "CP5", "CP6",
    "PZ", "P3", "P4", "P7", "P8", "PO3", "PO4", "OZ", "O1", "O2",
)


def main() -> None:
    base = Path(mne.__file__).parent / "data" / "fsaverage"
    scalp = mne.read_bem_surfaces(base / "fsaverage-head.fif", verbose=False)[0]
    fsaverage_head_to_mri = mne.read_trans(base / "fsaverage-trans.fif")
    mri_to_head = mne.transforms.invert_transform(fsaverage_head_to_mri)

    # fsaverage scalp is in MRI coordinates; display geometry is in head coordinates.
    scalp_mri = np.asarray(scalp["rr"], dtype=float)
    scalp_head = mne.transforms.apply_trans(mri_to_head, scalp_mri)

    # standard_1020's native positions are also MRI coordinates. Align its fiducials
    # to the head frame before finding a point on the fsaverage scalp surface.
    montage = mne.channels.make_standard_montage("standard_1020")
    montage_mri = montage.get_positions()["ch_pos"]
    montage_to_head = mne.channels.compute_native_head_t(montage)
    normalized_names = {name.upper(): name for name in montage_mri}

    electrodes = []
    for channel in CHANNELS:
        source_name = normalized_names[channel]
        source_mri = np.asarray(montage_mri[source_name], dtype=float)
        source_head = mne.transforms.apply_trans(montage_to_head, source_mri)
        nearest_index = int(np.argmin(np.sum((scalp_head - source_head) ** 2, axis=1)))
        display_head = scalp_head[nearest_index]
        distance_mm = float(np.linalg.norm(display_head - source_head) * 1000)
        electrodes.append({
            "name": channel,
            "position_head": np.round(display_head, 6).tolist(),
            "source_position_mri": np.round(source_mri, 6).tolist(),
            "scalp_distance_mm": round(distance_mm, 2),
        })

    result = {
        "source": "MNE-Python bundled fsaverage-head.fif and standard_1020 montage",
        "coordinate_note": "Mesh display in head frame; SH evaluation in original MRI/montage frame",
        "vertices": np.round(scalp_head, 6).tolist(),
        "sh_coordinates": np.round(scalp_mri, 6).tolist(),
        "triangles": np.asarray(scalp["tris"], dtype=int).tolist(),
        "electrodes": electrodes,
    }
    output = ROOT / "fsaverage_head_mesh.json"
    output.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    print(f"Saved {output}: {len(scalp_head)} vertices, {len(result['triangles'])} triangles")
    for electrode in electrodes:
        print(f"{electrode['name']:>3}  scalp distance {electrode['scalp_distance_mm']:5.2f} mm")


if __name__ == "__main__":
    main()
