"""CPU-only real TUEG optimizer-step check before submitting the GR2 run."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, type=Path)
    args = parser.parse_args()
    folder = args.folder.resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    entry = manifest["pretrain_entries"][0]
    source = Path(entry["source"])
    sys.path.insert(0, str(source))
    os.chdir(source)
    import torch
    import yaml
    from ablation.bootstrap import ensure_data_imports
    ensure_data_imports()
    from ablation.models import build_pretrain, parameter_report
    from ablation.sources import verify_sources
    from src.model import PretrainModel
    from src.modules.masking import make_masks
    from src.training.pretrain_rng import initialize_pretrain_rng
    torch.set_num_threads(2)
    config = yaml.safe_load(Path(entry["config"]).read_text())
    provenance = verify_sources()
    torch.manual_seed(config["seed"])
    model = build_pretrain(config, torch.device("cpu"))
    report = dict(parameters=parameter_report(model), sources=provenance)
    base = PretrainModel(config, torch.device("cpu"))
    base.load_state_dict({k.replace("backbone.encoder.core.", "backbone.encoder."): v
                          for k, v in model.state_dict().items()}, strict=True)
    report["base_model_strict_load"] = True
    # Adapter adds a module-name prefix, but must preserve full MJDE computation.
    signals = torch.randn(1, 19, 6000)
    shared_masks = make_masks(1, 19, 30, config["masking"], torch.device("cpu"), model.mask_coordinates)
    model.eval()
    base.eval()
    adapted = model(signals, shared_masks)[0]
    original = base(signals, shared_masks)[0]
    assert torch.equal(adapted, original)
    adapted.square().mean().backward()
    original.square().mean().backward()
    base_parameters = dict(base.named_parameters())
    for name, parameter in model.named_parameters():
        reference = base_parameters[name.replace("backbone.encoder.core.", "backbone.encoder.")]
        assert (parameter.grad is None) == (reference.grad is None), name
        if parameter.grad is not None:
            assert torch.equal(parameter.grad, reference.grad), name
    report["base_output_gradient_exact"] = True
    all_masks = []
    for rank in range(4):
        initialize_pretrain_rng(config, rank)
        masks = make_masks(128, 19, 30, config["masking"], torch.device("cpu"), model.mask_coordinates)
        assert (masks["target_mask"].sum((1, 2)) == 285).all()
        assert not (masks["target_mask"] & masks["context_mask"]).any()
        all_masks.append(masks["target_mask"])
    assert all(not torch.equal(all_masks[i], all_masks[j]) for i in range(4) for j in range(i))
    report["rank_masks_distinct"] = True
    report["target_tokens_per_sample"] = 285
    del model, base, all_masks
    subprocess.run([sys.executable, "-m", "ablation.pretrain", "--config", entry["config"],
                    "--device", "cpu", "--smoke"], check=True)
    output = source / "outputs/smoke" / (config["ablation"]["name"] + "_seed42")
    saved = torch.load(output / "last.pth", map_location="cpu", weights_only=False)
    assert saved["extra"]["partial_epoch_smoke"] is True
    assert saved["extra"]["step"] == 1
    restored = build_pretrain(config, torch.device("cpu"))
    restored.load_state_dict(saved["model"], strict=True)
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 1 and math.isfinite(metrics[0]["loss"])
    assert math.isfinite(metrics[0]["pre_clip_norm"])
    report.update(passed=True, real_data_step=metrics[0],
                  dataset_fingerprint=saved["extra"]["dataset_fingerprint"],
                  smoke_checkpoint=str(output / "last.pth"), partial_epoch_smoke=True,
                  permitted_as_training_resume=False)
    (folder / "validation/result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
