"""Summarize TUAB validation curves and five-seed results without changing selection."""

import argparse
import json
from pathlib import Path
import statistics

from ablation.tuab.campaign import ARMS, DEFAULT_CAMPAIGN, ROOT, SEEDS, write_json


def summary(values):
    return {"n": len(values), "mean": statistics.mean(values), "sd": statistics.pstdev(values)} if values else None


def report(campaign):
    manifest = json.loads((campaign / "manifest.json").read_text())
    arms = []
    for arm in manifest.get("arms", ARMS):
        runs = []
        for entry in manifest["entries"]:
            if entry["arm"] != arm:
                continue
            output = Path(entry["output"])
            history = {}
            if (output / "validation.jsonl").exists():
                for line in (output / "validation.jsonl").read_text().splitlines():
                    row = json.loads(line)
                    # An interrupted epoch may appear twice after resume; keep its last record.
                    history[row["epoch"]] = row["balanced_accuracy"]
            result = json.loads((output / "result.json").read_text()) if (output / "result.json").exists() else None
            early = [value for epoch, value in history.items() if epoch <= 5]
            late = [value for epoch, value in history.items() if epoch >= 6]
            run = {"seed": entry["seed"], "epochs_recorded": len(history), "complete": result is not None,
                   "early_best_val_bacc": max(early) if early else None,
                   "late_best_val_bacc": max(late) if late else None}
            if result:
                selected = result["balanced_accuracy"]
                run.update(best_epoch=selected["selection"]["epoch"],
                           selected_val_bacc=selected["selection"]["score"],
                           test_bacc=selected["test"]["balanced_accuracy"])
            runs.append(run)
        completed = [r for r in runs if r["complete"]]
        aggregate = {"arm": arm, "completed": len(completed), "total": len(SEEDS), "runs": runs,
                     "selected_val_bacc": summary([r["selected_val_bacc"] for r in completed]),
                     "test_bacc": summary([r["test_bacc"] for r in completed]),
                     "late_minus_early_val_bacc": summary([r["late_best_val_bacc"] - r["early_best_val_bacc"]
                         for r in completed if r["late_best_val_bacc"] is not None and r["early_best_val_bacc"] is not None])}
        arms.append(aggregate)
        print(json.dumps({k: v for k, v in aggregate.items() if k != "runs"}))
    write_json(campaign / "validation_summary.json", {"arms": arms,
               "interpretation": "Compare complete five-seed validation BAcc against the historical control. A later best epoch alone is not improvement."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=ROOT / DEFAULT_CAMPAIGN)
    report(parser.parse_args().campaign.resolve())
