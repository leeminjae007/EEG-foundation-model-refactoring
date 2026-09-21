from pathlib import Path

import yaml


SEEDS = (42, 696, 1001, 1234, 3407)


def test_mental_arithmetic_default_is_fixed_for_all_five_seeds():
    for seed in SEEDS:
        path = Path("configs/downstream") / f"gr9-1_mentalarithmetic_seed{seed}.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["seed"] == seed
        assert config["data"]["dataset"] == "stress"
        assert config["model"]["head_dropout"] == 0.1
        optimization = config["optimization"]
        assert optimization["weight_decay"] == 0.02
        assert {optimization[key] for key in (
            "tokenizer_learning_rate", "encoder_learning_rate", "head_learning_rate"
        )} == {1e-4}
        assert not any("warmup" in key.lower() for key in optimization)
