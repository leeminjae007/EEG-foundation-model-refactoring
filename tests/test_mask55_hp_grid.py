from pathlib import Path

import yaml

from scripts import submit_mask55_hp_grid as grid


def test_complete_requested_grid_and_fixed_recipe():
    assert len(grid.candidates("mentalarithmetic")) == 36
    assert len(grid.candidates("physionet_mi")) == 36
    mental = yaml.safe_load(Path("configs/downstream/gr9-1_mentalarithmetic_seed42.yaml").read_text())
    physio = yaml.safe_load(Path("configs/downstream/gr9-1_physionet_mi_seed42.yaml").read_text())
    checkpoint = Path("/tmp/checkpoint.pth")
    for dataset, base in (("mentalarithmetic", mental), ("physionet_mi", physio)):
        for hp in grid.candidates(dataset):
            config = grid.make_config(base, hp, checkpoint, Path("/tmp/output"))
            opt = config["optimization"]
            assert {opt[k] for k in ("tokenizer_learning_rate", "encoder_learning_rate", "head_learning_rate")} == {hp["learning_rate"]}
            assert opt["weight_decay"] == hp["weight_decay"]
            assert config["model"]["head_dropout"] == hp["dropout"]
            assert opt["epochs"] == 50 and opt["batch_size_per_gpu"] == 64
            assert config["model"]["head_hidden_tokens"] is None


def test_candidate_ids_are_unique():
    for dataset in grid.GRIDS:
        ids = {grid.candidate_id(hp) for hp in grid.candidates(dataset)}
        assert len(ids) == 36
