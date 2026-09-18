"""Rebuild round 01 after the missing round-00 task 22 completes."""

import argparse
import json
from pathlib import Path

from ablation.bciciv2a.campaign import (
    BEAM_WIDTH,
    MAX_CANDIDATES,
    candidate_id,
    load_plan,
    neighbors,
    plan_stage,
    rank_scores,
    read_scores,
    stage_dir,
    submit_stage,
    verify_source,
    write_json,
)


def reconcile(campaign: Path) -> None:
    verify_source(campaign)
    report = read_scores(campaign, "round_00")
    expected = len(load_plan(campaign, "round_00")["candidates"])
    if report["failures"] or len(report["scores"]) != expected:
        write_json(campaign / "task22_reconciliation.json", {
            "status": "round_00_incomplete", "expected_candidates": expected, **report})
        raise RuntimeError("Round 00 is still incomplete after task 22 retry")

    write_json(stage_dir(campaign, "round_00") / "scores.json", report)
    ranking = rank_scores(report["scores"])
    write_json(campaign / "ranking.json", ranking)
    seen = {candidate_id(hp) for hp in load_plan(campaign, "round_00")["candidates"]}
    candidates = neighbors([row["hp"] for row in ranking[:BEAM_WIDTH]], seen)
    candidates = candidates[:max(0, MAX_CANDIDATES - len(seen))]
    if not candidates:
        raise RuntimeError("Corrected round 00 produced no round-01 candidates")

    current = stage_dir(campaign, "round_01")
    archived = current.with_name("round_01_before_task22_recovery")
    reconciliation_path = campaign / "task22_reconciliation.json"
    previous_status = None
    if reconciliation_path.exists():
        previous_status = json.loads(reconciliation_path.read_text()).get("status")
    if current.exists() and previous_status != "corrected_round_01_ready":
        if archived.exists():
            raise RuntimeError("Refusing to replace an existing round-01 archive")
        if any(current.glob("runs/**/result.json")):
            raise RuntimeError("Refusing to archive a round 01 that has scientific results")
        current.rename(archived)

    plan = plan_stage(campaign, "round_01", candidates)
    write_json(reconciliation_path, {
        "status": "corrected_round_01_ready",
        "round_00_candidates": expected,
        "round_01_candidates": len(candidates),
        "archived_plan": str(archived),
        "ranking": [{"candidate": row["candidate"], "validation_mean": row["validation_mean"]}
                    for row in ranking],
        "plan_entries": len(plan["entries"]),
    })
    submit_stage(campaign, "round_01")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    reconcile(parser.parse_args().campaign.resolve())
