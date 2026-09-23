import json
import subprocess

import yaml

from scripts import submit_mask55_final_comparison as final


def test_scope_and_resource_routing():
    assert len(final.DATASETS) == 10
    assert "tuab" not in final.DATASETS and "tusl" not in final.DATASETS
    assert set(final.ARMS["hk4935"]) == {"pe-ch_order", "pe-acpe", "pe-4dREVE"}
    assert "enc-s2t-6stage" not in final.ARMS["yc8820"]
    assert "enc-t2s-6stage" not in final.ARMS["yc8820"]
    assert final.HP["mentalarithmetic"] == (1e-4, .02, .1)
    assert final.HP["physionet_mi"] == (1e-4, .02, .1)
    assert final.resources("a100", "chb")["partitions"] == "a100_short,a100_long"
    assert final.resources("gl40s", "tusz")["partitions"] == "gl40s_dev,gl40s_short,gl40s_long"
    assert "gl40s-8013" in final.resources("gl40s", "hmc")["excluded_nodes"]


def test_reuse_requires_five_seed_result_and_exact_default(tmp_path):
    old = tmp_path / "old"
    seed = 42
    config = old / f"configs/downstream/physionet_mi_seed{seed}.yaml"
    result = old / f"downstream/physionet_mi/seed-{seed}/result.json"
    config.parent.mkdir(parents=True)
    result.parent.mkdir(parents=True)
    hp = final.HP["physionet_mi"]
    values = dict(optimization=dict(tokenizer_learning_rate=hp[0],
                                    encoder_learning_rate=hp[0],
                                    head_learning_rate=hp[0],
                                    weight_decay=hp[1]),
                  model=dict(head_dropout=hp[2]))
    config.write_text(yaml.safe_dump(values))
    result.write_text(json.dumps(dict(balanced_accuracy=dict(
        selection=dict(score=.6), test=dict(balanced_accuracy=.65)))))
    assert final.matching_old(old, "physionet_mi", seed)
    values["optimization"]["encoder_learning_rate"] = 5e-5
    config.write_text(yaml.safe_dump(values))
    assert not final.matching_old(old, "physionet_mi", seed)
    values["optimization"]["encoder_learning_rate"] = hp[0]
    config.write_text(yaml.safe_dump(values))
    result.write_text("{")
    assert not final.matching_old(old, "physionet_mi", seed)


def test_purged_slurm_id_is_not_an_active_job(monkeypatch):
    monkeypatch.setattr(final.subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess(a[0], 1, "", "slurm_load_jobs error: Invalid job id specified"))
    assert final.slurm_active_state("27678704") == ""
