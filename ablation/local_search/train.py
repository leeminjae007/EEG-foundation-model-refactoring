"""Run a verified candidate, or evaluate a previously locked five-seed winner."""
import argparse
from pathlib import Path
from types import SimpleNamespace

from ablation.local_search.campaign import load_plan, read, write_json, check_result


def run(campaign, slug, stage, index):
    from src.training.runtime import set_paths
    set_paths()
    import torch
    import yaml
    from src.training import engine
    plan = load_plan(campaign, slug, stage)
    e = plan["entries"][index]
    config = yaml.safe_load(Path(e["config"]).read_text())
    output = Path(e["output"])
    output.mkdir(parents=True, exist_ok=True)
    if stage != "evaluate":
        checkpoint = output / "last.pth"
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu")
            if saved["config"] != config or saved.get("extra", {}).get("partial_epoch_smoke"):
                raise ValueError("Resume requires the identical full-run config")
            del saved
        (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        try:
            engine.run_finetune(config, SimpleNamespace(device="cuda", distributed=True, smoke=False,
                                resume=str(checkpoint) if checkpoint.exists() else None))
        except RuntimeError as exc:
            if "non-finite" not in str(exc).lower() and "nonfinite" not in str(exc).lower():
                raise
            write_json(output / "pruned.json", dict(reason="nonfinite_gradient", error=str(exc),
                                                    completed=False, seed=e["seed"]))
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            return
        check_result(e)
        return
    # No test data can be opened before the three-seed validation decision is persisted.
    winner = read(campaign / "datasets" / slug / "winner.json")
    if not winner["locked_before_test"] or winner["hp"] != e["hp"]:
        raise ValueError("Winner was not fixed before test evaluation")
    if sorted(plan["seeds"]) != [42, 696, 1001, 1234, 3407] or len(plan["entries"]) != 5:
        raise ValueError("Final evaluation needs all five seeds")
    for item in plan["entries"]:
        origin = item["origin"]
        r = read(Path(origin["output"]) / "result.json")
        if r["balanced_accuracy"]["selection"]["score"] != origin["score"]:
            raise ValueError("Selected validation result changed")
    origin = e["origin"]
    device, rank, world = engine.setup("cuda", True, config["seed"], False, False, True)
    spec = engine.get_dataset_spec(config["data"]["dataset"])
    dataset = spec.dataset_class(config["data"]["dataset_dir"], "test")
    dataset.enable_coordinate_only_channels()
    loader = torch.utils.data.DataLoader(dataset, batch_size=config["optimization"]["batch_size_per_gpu"],
        sampler=range(rank, len(dataset), world), num_workers=config["data"]["num_workers"],
        pin_memory=True, drop_last=False)
    if hasattr(engine, "preserve_loader_worker_affinity"):
        engine.preserve_loader_worker_affinity(loader)
    model = engine.build_finetune(config).to(device)
    model.load_state_dict(torch.load(Path(origin["output"]) / "best-balanced_accuracy.pth", map_location=device), strict=True)
    metrics = engine.evaluate(model, loader, spec.task, config["data"]["dataset"], device, world)
    if rank == 0:
        old = read(Path(origin["output"]) / "result.json")["balanced_accuracy"]
        write_json(output / "result.json", {"balanced_accuracy": {"selection": old["selection"], "test": metrics}})
        write_json(output / "provenance.json", dict(training_origin=origin, winner=str(campaign / "datasets" / slug / "winner.json")))
        evaluation_config = dict(config, runtime=dict(config["runtime"], evaluate_test=True))
        (output / "resolved_config.yaml").write_text(yaml.safe_dump(evaluation_config, sort_keys=False))
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", required=True, type=Path)
    p.add_argument("--slug", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--index", required=True, type=int)
    a = p.parse_args()
    run(a.campaign, a.slug, a.stage, a.index)
