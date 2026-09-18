"""Read completed, validated seed results without advancing an Optuna campaign."""
import argparse
import json
from pathlib import Path

from ablation.optuna_search.campaign import NAMES, read, validate_seed


def collect(campaign):
    rows = []
    for slug in NAMES:
        for plan_path in sorted((campaign / "datasets" / slug).glob("trial-*/plan.json")):
            plan = read(plan_path)
            for entry in plan["entries"]:
                if not (Path(entry["output"]) / "completed.json").is_file():
                    continue
                seed = validate_seed(entry)
                rows.append(dict(dataset=slug, trial=plan["number"], hp=plan["hp"],
                                 seed=seed["seed"], selected_epoch=seed["selection"]["epoch"],
                                 validation_bacc=seed["selection"]["score"],
                                 test=seed["test"], source_result=seed["source_result"]))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(collect(args.campaign), ensure_ascii=False))
