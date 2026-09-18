"""Read-only summary of validation-only local-search screening results."""

from __future__ import annotations

import json
from pathlib import Path


CAMPAIGN = Path(__file__).resolve().parents[1] / "outputs/knn37_local_search_20260916_1157"
ORDER = ("tuev", "tuab", "mentalarithmetic", "isruc", "hmc")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    manifest = read(CAMPAIGN / "manifest.json")
    for slug in ORDER:
        dataset = manifest["datasets"][slug]
        baseline = next(entry for entry in dataset["baseline"]
                        if int(entry["seed"]) == int(dataset["start_seed"]))
        plan = read(CAMPAIGN / "datasets" / slug / "screen/plan.json")
        rows = []
        for entry in plan["entries"]:
            result_path = Path(entry["output"]) / "result.json"
            if not result_path.exists():
                continue
            result = read(result_path)
            rows.append({"hp": entry["hp"], "candidate": entry["candidate"],
                         "validation_balanced_accuracy": result["balanced_accuracy"]["selection"]["score"],
                         "selected_epoch": result["balanced_accuracy"]["selection"]["epoch"]})
        rows.sort(key=lambda row: -row["validation_balanced_accuracy"])
        print(json.dumps({"dataset": slug, "stage": "single-seed screen",
                          "seed": dataset["start_seed"], "baseline_hp": dataset["anchor"],
                          "baseline_validation_balanced_accuracy": baseline["validation"],
                          "completed": len(rows), "planned": len(plan["entries"]),
                          "rows": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
