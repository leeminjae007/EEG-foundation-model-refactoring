import pytest

from scripts.downstream_gpu_guard import allocated_gpu


def test_uses_uuid_from_allocated_device_not_physical_ordinal():
    result = allocated_gpu("GPU-allocated, 45456, 44000\n", 12288)
    assert result["gpu_uuid"] == "GPU-allocated"
    assert result["free_mib"] == 44000


@pytest.mark.parametrize("text", ["", "GPU-a, 45456, 44000\nGPU-b, 45456, 44000", "0, 45456, 44000"])
def test_rejects_ambiguous_or_missing_gpu_identity(text):
    with pytest.raises(ValueError):
        allocated_gpu(text, 12288)


def test_rejects_observed_occupied_device_before_loading_training():
    with pytest.raises(RuntimeError, match="insufficient free memory"):
        allocated_gpu("GPU-occupied, 45456, 6000", 12288)


def test_local_sync_cannot_advance_optuna(monkeypatch):
    from types import SimpleNamespace
    from scripts import monitor_optuna_search as monitor

    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout='{"status": "running"}\n')

    monkeypatch.setattr(monitor.subprocess, "run", run)
    args = SimpleNamespace(root="/repo", campaign="/repo/campaign", read_only=True,
                           host="cluster", fallback_hosts=[])
    assert monitor.fetch(args)["status"] == "running"
    assert "status.json" in calls[0][-1]
    assert "ablation.optuna_search.campaign" not in calls[0][-1]
