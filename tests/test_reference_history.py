"""Follow-up 변경 전 소스와 현재 소스의 GR9-1 경로가 같은지 확인한다."""

import json
import os
import subprocess
import sys

import torch
from src.training.runtime import ROOT


def test_followup_source_did_not_change_gr9_1():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run([sys.executable, str(ROOT / "tests/reference_before/export.py")],
                   cwd=ROOT / "tests/reference_before", env=env, check=True)
    before = torch.load(ROOT / "outputs/oracle_before.pth", map_location="cpu")
    current = torch.load(ROOT / "outputs/oracle.pth", map_location="cpu")
    maximum = 0.0
    for key in ("signals", "target", "context", "prediction", "loss", "train_prediction", "train_loss"):
        delta = float((before[key] - current[key]).abs().max())
        maximum = max(maximum, delta)
        torch.testing.assert_close(before[key], current[key], rtol=0, atol=0, msg=key)
    for key, value in before["gradients"].items():
        delta = float((value - current["gradients"][key]).abs().max())
        maximum = max(maximum, delta)
        torch.testing.assert_close(value, current["gradients"][key], rtol=0, atol=0, msg=key)
    assert torch.equal(before["rng_after_train"], current["rng_after_train"])
    report = {"before_archive": "outputs/gr9_followup_20260912/source_before.tgz",
              "compared_with": "current source copied at refactor start",
              "output_loss_gradient_max_abs": maximum,
              "dropout_final_rng_equal": True,
              "does_not_prove_original_run_source_hash": True}
    (ROOT / "outputs/source-history-equivalence.json").write_text(json.dumps(report, indent=2))
