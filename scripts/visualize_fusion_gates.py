"""Create an auditable visualization of the learned GR2 fusion gates."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def encoder_state(checkpoint):
    state = {}
    for key, value in checkpoint["model"].items():
        if key.startswith("backbone.encoder.core."):
            state["encoder." + key[len("backbone.encoder.core."):]] = value
        elif key.startswith("backbone.encoder."):
            state[key[len("backbone."):]] = value
    return state


def gate_mode(checkpoint):
    """Resolve gate mode for either a pretrain or downstream checkpoint."""
    encoder = checkpoint.get("config", {}).get("encoder")
    if encoder is not None:
        return encoder.get("fusion_gate", "static_feature")
    # Finetune configs intentionally contain only task-specific settings; their
    # model.checkpoint points to the pretrain artifact that defines the encoder.
    pretrain_path = checkpoint.get("config", {}).get("model", {}).get("checkpoint")
    if not pretrain_path:
        raise ValueError("checkpoint has no encoder config or pretrain checkpoint reference")
    pretrain = torch.load(pretrain_path, map_location="cpu")
    return pretrain["config"]["encoder"].get("fusion_gate", "static_feature")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="fusion_gate",
                        help="artifact prefix; use a distinct value for downstream final gates")
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu")
    state = encoder_state(saved)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mode = gate_mode(saved)
    report = {"checkpoint": str(args.checkpoint), "mode": mode, "artifacts": []}
    if mode == "static_feature":
        gates = state["encoder.fusion_gates"].sigmoid().numpy()
        fig, axis = plt.subplots(figsize=(11, 3.5))
        image = axis.imshow(gates, aspect="auto", vmin=0, vmax=1, cmap="coolwarm")
        axis.set(xlabel="feature dimension", ylabel="fusion stage", yticks=range(3), yticklabels=["stage 1", "stage 2", "stage 3"], title="Static fusion gate: S→T weight")
        fig.colorbar(image, ax=axis, label="sigmoid(gate)")
        path = args.output_dir / (args.prefix + "_static.png"); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
        report.update(stage_mean=gates.mean(axis=1).tolist(), stage_sd=gates.std(axis=1).tolist(), artifacts=[str(path)])
    else:
        rows = []
        for stage in range(3):
            weight = state["encoder.patch_gates.%d.weight" % stage].float()
            dim = weight.shape[1] // 2
            rows.append(torch.stack((weight[:, :dim].norm(dim=1), weight[:, dim:].norm(dim=1))).numpy())
        values = torch.cat([torch.tensor(row) for row in rows], dim=0).numpy()
        fig, axis = plt.subplots(figsize=(8, 4))
        image = axis.imshow(values, aspect="auto", cmap="viridis")
        axis.set(xlabel="input route (S→T, T→S)", ylabel="stage × gate output", xticks=[0, 1], xticklabels=["S→T", "T→S"], title="Patch-gate learned coefficient norms")
        fig.colorbar(image, ax=axis, label="L2 norm")
        path = args.output_dir / (args.prefix + "_patch_coefficients.png"); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
        report.update(stage_route_norms=[row.mean(axis=1).tolist() for row in rows], artifacts=[str(path)])
    (args.output_dir / (args.prefix + "_report.json")).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
