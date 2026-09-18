import pytest
import torch

from ablation.config import load_config, resolve_ablation
from ablation.integration import connected_engine
from ablation.models import build_pretrain


def write_lmdb(path, records, keys):
    import lmdb
    import pickle
    path.mkdir()
    database = lmdb.open(str(path), map_size=32 * 1024 * 1024)
    with database.begin(write=True) as transaction:
        transaction.put(b"__keys__", pickle.dumps(keys))
        for key, value in records.items():
            transaction.put(key.encode(), pickle.dumps(value))
    database.close()


@pytest.mark.parametrize("arm", ["encoder_labram", "encoder_cbramod", "encoder_csbrain", "encoder_mjde",
                                "encoder_mjde_lite", "encoder_mjde_s2t6", "encoder_mjde_t2s6",
                                "encoder_mjde_average", "pe_none", "pe_channel_id", "pe_acpe",
                                "pe_reve4d", "pe_shpe"])
def test_pretrain_to_downstream_strict_loading_and_optimizer(tmp_path, arm):
    settings = resolve_ablation(load_config("ablation/configs/" + arm + ".yaml"))
    if "depth" in settings["ablation"]:
        settings["ablation"]["depth"] = 1
    pretrained = build_pretrain(settings, torch.device("cpu"))
    path = tmp_path / "pretrain.pth"
    torch.save({"model": pretrained.state_dict(), "config": settings}, path)
    downstream = load_config("configs/downstream/gr9-1_seedv_seed42.yaml")
    downstream["model"]["checkpoint"] = str(path)
    with connected_engine() as engine:
        model = engine.build_finetune(downstream)
        spec = engine.get_dataset_spec("seed-v")
        from src.data.electrode_geometry import resolve_channel_coordinates
        coordinates, valid = resolve_channel_coordinates(spec.dataset_class.channel_names)
        logits = model(torch.randn(2, 62, 200), coordinates[None].expand(2, -1, -1), valid[None].expand(2, -1))
        assert logits.shape == (2, 5)
        logits.square().mean().backward()
        parameters = [p for group in engine.optimizer_groups(model, downstream["optimization"]) for p in group["params"]]
        assert len(parameters) == len({id(p) for p in parameters})
        assert {id(p) for p in parameters} == {id(p) for p in model.parameters()}
        assert all(p.grad is not None for p in parameters if p.requires_grad)


def test_sleep_head_keeps_sequence_dimension(tmp_path):
    settings = resolve_ablation(load_config("ablation/configs/encoder_csbrain.yaml"))
    settings["ablation"]["depth"] = 1
    pretrained = build_pretrain(settings, torch.device("cpu"))
    path = tmp_path / "pretrain.pth"
    torch.save({"model": pretrained.state_dict(), "config": settings}, path)
    downstream = load_config("configs/downstream/gr9-1_isruc_seed42.yaml")
    downstream["model"]["checkpoint"] = str(path)
    with connected_engine() as engine:
        model = engine.build_finetune(downstream).eval()
        spec = engine.get_dataset_spec("isruc")
        from src.data.electrode_geometry import resolve_channel_coordinates
        coordinates, valid = resolve_channel_coordinates(spec.dataset_class.channel_names)
        with torch.no_grad():
            logits = model(torch.randn(1, 2, 6, 6000), coordinates[None], valid[None])
        assert logits.shape == (1, 2, 5)


def test_real_engine_pretrain_resume_and_downstream_evaluation(tmp_path, monkeypatch):
    """Exercise the unmodified data loader, optimizer, scheduler and checkpoint IO."""
    import json
    from types import SimpleNamespace
    import numpy as np

    generator = np.random.default_rng(8)
    training_data = tmp_path / "pretrain_data"
    records = {str(i): generator.standard_normal((19, 6, 200)).astype("float32") for i in range(4)}
    write_lmdb(training_data, records, list(records))
    settings = resolve_ablation(load_config("ablation/configs/encoder_cbramod.yaml"))
    settings["ablation"]["depth"] = 1
    settings["data"].update(dataset_dir=str(training_data), num_workers=0, num_patches=6, pin_memory=False)
    settings["optimization"].update(epochs=2, batch_size_per_gpu=2)
    settings["runtime"]["output"] = str(tmp_path / "uninterrupted")
    args = SimpleNamespace(device="cpu", distributed=False, smoke=False, resume=None)

    with connected_engine() as engine:
        engine.run_pretrain(settings, args)
        expected = torch.load(tmp_path / "uninterrupted/last.pth", weights_only=False)
        settings["runtime"]["output"] = str(tmp_path / "resumed")
        original_save = engine.save_checkpoint

        def stop_after_checkpoint(*values, **kwargs):
            original_save(*values, **kwargs)
            raise RuntimeError("simulated interruption")

        with monkeypatch.context() as patch:
            patch.setattr(engine, "save_checkpoint", stop_after_checkpoint)
            with pytest.raises(RuntimeError, match="simulated interruption"):
                engine.run_pretrain(settings, args)
        args.resume = str(tmp_path / "resumed/last.pth")
        engine.run_pretrain(settings, args)
        actual = torch.load(args.resume, weights_only=False)
        assert actual["epoch"] == 2
        assert actual["extra"]["step"] == expected["extra"]["step"] == 4
        for name, value in actual["model"].items():
            torch.testing.assert_close(value, expected["model"][name], rtol=0, atol=0)

        downstream_data = tmp_path / "downstream_data"
        splits, records = {}, {}
        for split in ("train", "val", "test"):
            keys = []
            for index in range(10):
                key = "subject" + str(index // 5) + "_" + split + "-" + str(index)
                keys.append(key)
                records[key] = {"sample": generator.standard_normal((62, 1, 200)).astype("float32"),
                                "label": index % 5}
            splits[split] = keys
        # Separate fixture environments allow num_workers=0 on Windows LMDB,
        # which disallows opening one environment for several splits in-process.
        downstream_data.mkdir()
        for split in splits:
            write_lmdb(downstream_data / split, records, splits)
        from dataclasses import replace
        original_spec = engine.get_dataset_spec("seed-v")

        class PerSplitDataset(original_spec.dataset_class):
            def __init__(self, directory, split):
                from pathlib import Path
                super().__init__(Path(directory) / split, split)

        test_spec = replace(original_spec, dataset_class=PerSplitDataset)
        monkeypatch.setattr(engine, "get_dataset_spec", lambda name: test_spec)
        downstream = load_config("configs/downstream/gr9-1_seedv_seed42.yaml")
        downstream["model"]["checkpoint"] = str(tmp_path / "resumed/last.pth")
        downstream["data"].update(dataset_dir=str(downstream_data), num_workers=0)
        downstream["optimization"].update(epochs=1, batch_size_per_gpu=5)
        downstream["runtime"]["output"] = str(tmp_path / "downstream")
        args.resume = None
        engine.run_finetune(downstream, args)
        result = json.loads((tmp_path / "downstream/result.json").read_text())
        assert set(result) == {"balanced_accuracy", "kappa"}
        assert all(item["selection"]["epoch"] == 1 for item in result.values())
