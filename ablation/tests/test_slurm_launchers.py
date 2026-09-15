"""Exercise submission and per-GPU rank mapping without a Slurm installation."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "ablation/scripts"
BASH = shutil.which("bash")
if not BASH and Path("C:/Program Files/Git/bin/bash.exe").exists():
    BASH = "C:/Program Files/Git/bin/bash.exe"
pytestmark = pytest.mark.skipif(not BASH, reason="Bash is required for Slurm launcher checks")


def bash_path(path):
    path = Path(path).resolve().as_posix()
    return "/" + path[0].lower() + path[2:] if os.name == "nt" else path


def stub(folder, name, body):
    path = folder / name
    with path.open("w", encoding="utf-8", newline="\n") as output:
        output.write("#!/usr/bin/env bash\nset -euo pipefail\n" + body + "\n")
    path.chmod(0o755)


@pytest.fixture
def shell(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    events = tmp_path / "events"
    env = dict(os.environ, MOCK_BIN=bash_path(binaries), EVENTS=bash_path(events), USER="ablation-test")

    def run(script, *args, extra_env=None, check=True):
        result = subprocess.run(
            [BASH, "-c", 'export PATH="$MOCK_BIN:$PATH"; exec bash "$@"', "launcher-test", str(script), *args],
            cwd=ROOT, env=dict(env, **(extra_env or {})), capture_output=True, text=True, timeout=30,
        )
        if check:
            assert result.returncode == 0, result.stdout + result.stderr
        return result

    return binaries, events, run


def test_encoder_submission_and_existing_jobs(shell):
    binaries, events, run = shell
    stub(binaries, "squeue", "exit 0")
    stub(binaries, "sbatch", 'printf "%s\\n" "$*" >> "$EVENTS"')
    run(SCRIPTS / "run_pretrain.sh", "encoder", "42")
    submitted = events.read_text().splitlines()
    assert len(submitted) == 5
    for line, arm in zip(submitted, ("labram", "cbramod", "csbrain", "mjde", "mjde_lite")):
        assert "--job-name=enc-" + arm + "-s42" in line
        assert "pretrain_flexible.slurm encoder_" + arm + " 42" in line

    events.unlink()
    stub(binaries, "squeue", 'printf "101 RUNNING\\n102 PENDING\\n"')
    result = run(SCRIPTS / "run_pretrain.sh", "encoder_labram")
    assert "Skip enc-labram-s42" in result.stdout
    assert not events.exists()


def test_replace_pending_preserves_running_and_racing_jobs(shell, tmp_path):
    binaries, events, run = shell
    state = tmp_path / "cancelled"
    stub(binaries, "squeue", '[[ -f "$STATE" ]] || printf "101 PENDING\\n"')
    stub(binaries, "scancel", 'printf "cancel %s\\n" "$*" >> "$EVENTS"; touch "$STATE"')
    stub(binaries, "sbatch", 'printf "submit %s\\n" "$*" >> "$EVENTS"')
    run(SCRIPTS / "run_pretrain.sh", "--replace-pending", "encoder_labram", extra_env={"STATE": bash_path(state)})
    lines = events.read_text().splitlines()
    assert lines[0] == "cancel --state=PENDING 101"
    assert lines[1].startswith("submit --job-name=enc-labram-s42")

    events.unlink()
    stub(binaries, "squeue", 'printf "101 RUNNING\\n"')
    run(SCRIPTS / "run_pretrain.sh", "--replace-pending", "encoder_labram")
    assert not events.exists()

    state.unlink()
    stub(binaries, "squeue", 'if [[ -f "$STATE" ]]; then printf "101 RUNNING\\n"; else printf "101 PENDING\\n"; fi')
    run(SCRIPTS / "run_pretrain.sh", "--replace-pending", "encoder_labram", extra_env={"STATE": bash_path(state)})
    assert events.read_text().splitlines() == ["cancel --state=PENDING 101"]


def test_reve_and_resume_arguments(shell):
    binaries, events, run = shell
    stub(binaries, "squeue", "exit 0")
    stub(binaries, "sbatch", 'printf "<%s>\\n" "$@" >> "$EVENTS"')
    run(SCRIPTS / "run_reve.sh", "--replace-pending", "--resume", "outputs/a run/last.pth")
    arguments = events.read_text().splitlines()
    assert "<--job-name=pe-reve4d-s42>" in arguments
    assert arguments[-4:] == ["<pe_reve4d>", "<42>", "<--resume>", "<outputs/a run/last.pth>"]


@pytest.mark.parametrize("local_ids", [(0, 1, 2, 3), (0, 1, 0, 1), (0, 0, 0, 0), (0, 1, 0, 0)])
def test_each_gpu_gets_unique_global_rank_and_cuda_zero(shell, tmp_path, local_ids):
    binaries, events, run = shell
    stub(binaries, "python", 'printf "%s|%s|%s|%s|%s|%s\\n" "$RANK" "$WORLD_SIZE" "$LOCAL_RANK" "$CUDA_VISIBLE_DEVICES" "$MASTER_ADDR" "$*" >> "$EVENTS"')
    for rank, local_id in enumerate(local_ids):
        run(SCRIPTS / "slurm_worker.sh", "--distributed", extra_env={
            "SLURM_PROCID": str(rank), "SLURM_LOCALID": str(local_id), "SLURM_NTASKS": "4",
            "SLURM_JOB_ID": "1234", "SLURM_TMPDIR": bash_path(tmp_path),
            "MASTER_ADDR": "gpu-node01", "MASTER_PORT": "23456", "CUDA_VISIBLE_DEVICES": str(local_id + 2),
            "RANK": "99", "WORLD_SIZE": "99", "LOCAL_RANK": "99",
        })
    rows = [line.split("|") for line in events.read_text().splitlines()]
    assert [row[0] for row in rows] == ["0", "1", "2", "3"]
    assert all(row[1:3] == ["4", "0"] for row in rows)
    assert all(row[4:] == ["gpu-node01", "-u -m ablation.pretrain --distributed"] for row in rows)
    assert (tmp_path / "eeg-ablation-1234").is_dir()


@pytest.mark.parametrize("visible", ["", "0,1", "-1"])
def test_worker_rejects_missing_or_shared_gpu_binding(shell, visible):
    _, _, run = shell
    result = run(SCRIPTS / "slurm_worker.sh", extra_env={
        "SLURM_PROCID": "0", "SLURM_NTASKS": "4", "SLURM_JOB_ID": "1234",
        "MASTER_ADDR": "gpu-node01", "MASTER_PORT": "23456", "CUDA_VISIBLE_DEVICES": visible,
    }, check=False)
    assert result.returncode == 2
    assert "Expected one visible GPU per task" in result.stderr


def test_batch_starts_one_task_per_gpu_with_shared_rendezvous(shell, tmp_path):
    binaries, events, run = shell
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "scripts/activate.sh").write_text("# Mock the existing server virtualenv activation.\n")
    stub(binaries, "scontrol", 'printf "gpu-node01\\ngpu-node02\\ngpu-node03\\n"')
    stub(binaries, "srun", 'printf "%s:%s\\n" "$MASTER_ADDR" "$MASTER_PORT" >> "$EVENTS"; printf "<%s>\\n" "$@" >> "$EVENTS"')
    run(SCRIPTS / "pretrain_flexible.slurm", "encoder_labram", "42", "--output", "a run", extra_env={
        "SLURM_SUBMIT_DIR": bash_path(project), "SLURM_JOB_NODELIST": "gpu-node[01-03]",
        "SLURM_JOB_ID": "1234", "SLURM_JOB_NUM_NODES": "3", "SLURM_NTASKS": "4",
        "ABLATION_MASTER_PORT": "24321",
    })
    lines = events.read_text().splitlines()
    assert lines[0] == "gpu-node01:24321"
    for option in ("--nodes=3", "--ntasks=4", "--gpus-per-task=a100:1", "--gpu-bind=single:1", "--kill-on-bad-exit=1"):
        assert "<" + option + ">" in lines
    assert lines[-2:] == ["<--output>", "<a run>"]


def test_queue_failure_does_not_submit_duplicates(shell):
    binaries, events, run = shell
    stub(binaries, "squeue", "exit 1")
    stub(binaries, "sbatch", 'touch "$EVENTS"')
    result = run(SCRIPTS / "run_pretrain.sh", "encoder_labram", check=False)
    assert result.returncode != 0
    assert not events.exists()
