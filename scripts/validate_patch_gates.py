"""CPU-only real-data smoke checks; these are not downstream performance results."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source))
    import torch
    import yaml
    from src.model import PretrainModel
    from src.modules.masking import make_masks
    from src.training.diagnostics import capture, measurements
    from src.training.engine import run_pretrain
    from src.training.runtime import set_paths
    from src.data.datasets.pretraining_dataset import make_pretraining_loader
    set_paths()
    torch.set_num_threads(2)
    reference_config = yaml.safe_load((source / "configs/pretrain_gr2_geometry.yaml").read_text())
    torch.manual_seed(42)
    baseline = PretrainModel(reference_config, torch.device("cpu"))
    baseline_state = baseline.state_dict()
    baseline_rng = torch.get_rng_state()
    reports = {}
    for mode in ("patch_scalar", "patch_feature"):
        config = yaml.safe_load((source / ("configs/pretrain_gr2_" + mode + ".yaml")).read_text())
        torch.manual_seed(42)
        initialized = PretrainModel(config, torch.device("cpu"))
        assert torch.equal(torch.get_rng_state(), baseline_rng)
        for name, value in initialized.state_dict().items():
            assert torch.count_nonzero(value) == 0 if ".patch_gates." in name else torch.equal(value, baseline_state[name]), name
        report = dict(gate_parameters=sum(p.numel() for p in initialized.backbone.encoder.patch_gates.parameters()),
                      encoder_parameters=sum(p.numel() for p in initialized.backbone.encoder.parameters()),
                      total_parameters=sum(p.numel() for p in initialized.parameters()),
                      initial_common_weights_exact=True, initial_rng_exact=True)
        del initialized
        run_pretrain(config, SimpleNamespace(device="cpu", distributed=False, smoke=True, resume=None))
        smoke = source / "outputs/smoke" / Path(config["runtime"]["output"]).name
        saved = torch.load(smoke / "last.pth", map_location="cpu", weights_only=False)
        assert saved["extra"]["partial_epoch_smoke"] and saved["extra"]["step"] == 1
        model = PretrainModel(config, torch.device("cpu")).eval()
        model.load_state_dict(saved["model"], strict=True)
        training_metrics = json.loads((smoke / "metrics.jsonl").read_text().splitlines()[0])
        assert torch.isfinite(torch.tensor(training_metrics["loss"]))
        gate_gradients = {k: v for k, v in training_metrics.items() if "/gate/" in k and "gradient_rms" in k}
        assert len(gate_gradients) == 6 and all(v > 0 for v in gate_gradients.values())
        assert all(p.abs().sum() > 0 for p in model.backbone.encoder.patch_gates.parameters())
        data = config["data"]
        dataset, loader, sampler = make_pretraining_loader(dataset_dir=data["dataset_dir"], batch_size=2,
            pin_mem=False, num_workers=0, world_size=1, rank=0, seed=42, drop_last=True,
            channels=len(data["channel_names"]), num_patches=data["num_patches"],
            patch_samples=config["patch_encoder"]["patch_samples"])
        signals = next(iter(loader))[0] / data["value_scale"]
        torch.manual_seed(42)
        masks = make_masks(2, 19, 30, config["masking"], torch.device("cpu"), model.mask_coordinates)
        capture(model, True)
        with torch.no_grad():
            model(signals, masks)
        gate_values = {k: v for k, v in measurements(model).items() if "/gate/" in k}
        variation = [v for k, v in gate_values.items() if k.endswith("gate/patch_std_mean")]
        assert len(variation) == 3 and all(v > 0 for v in variation)
        if mode == "patch_feature":
            assert all(v > 0 for k, v in gate_values.items() if k.endswith("gate/feature_std_mean"))
        report.update(passed=True, loss=training_metrics["loss"], pre_clip_norm=training_metrics["pre_clip_norm"],
                      dataset_fingerprint=saved["extra"]["dataset_fingerprint"], gate_gradients=gate_gradients,
                      gate_statistics_after_one_step=gate_values, strict_checkpoint_load=True,
                      smoke_checkpoint=str(smoke / "last.pth"), partial_epoch_smoke=True)
        reports[mode] = report
        dataset.close()
        del dataset, loader, model, saved
        (output / "validation_progress.json").write_text(json.dumps(reports, indent=2))
        print(json.dumps({mode: report}), flush=True)
    assert reports["patch_scalar"]["loss"] == reports["patch_feature"]["loss"]
    result = dict(passed=True, validation_only=True, optimizer_steps_per_variant=1,
                  first_step_loss_exact=True, variants=reports)
    (output / "validation_result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
