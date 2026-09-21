from pathlib import Path

import yaml

from scripts import submit_mask55_clinical_hp_grid as grid


def test_grid_shape_and_unique_candidates():
    assert len(grid.candidates()) == 36
    assert len({grid.candidate_id(hp) for hp in grid.candidates()}) == 36
    assert len(grid.DATASETS) * len(grid.candidates()) * len(grid.SEEDS) == 540


def test_only_requested_hyperparameters_change():
    for dataset in grid.DATASETS:
        path = Path("configs/downstream") / f"gr9-1_{dataset}_seed42.yaml"
        base = yaml.safe_load(path.read_text())
        for hp in grid.candidates():
            result = grid.make_config(base, hp, Path("/tmp/checkpoint.pth"), Path("/tmp/output"))
            assert not any("warmup" in str(key).lower() for key in result["optimization"])
            assert {result["optimization"][key] for key in
                    ("tokenizer_learning_rate", "encoder_learning_rate", "head_learning_rate")} == {
                        hp["learning_rate"]}
            assert result["optimization"]["weight_decay"] == hp["weight_decay"]
            assert result["model"]["head_dropout"] == hp["dropout"]
            for key in ("epochs", "batch_size_per_gpu", "gradient_accumulation_steps",
                        "label_smoothing", "class_counts"):
                assert result["optimization"].get(key) == base["optimization"].get(key)
