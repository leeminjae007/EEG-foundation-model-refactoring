import copy
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import launch_tuab_handoff as handoff
from scripts import audit_tuab_handoff as audit
from scripts import monitor_experiment_results as publisher


@pytest.mark.parametrize('state', sorted(handoff.ACTIVE))
def test_active_seeds_never_get_resubmitted(state):
    assert handoff.decision(state, False) == 'reuse_running'
    assert handoff.decision(state, True) == 'reuse_running'


@pytest.mark.parametrize('state', ['PENDING', None, 'UNKNOWN'])
def test_uncertain_or_pending_jobs_refuse_handoff(state):
    with pytest.raises(ValueError):
        handoff.decision(state, False)


@pytest.mark.parametrize('state', ['CANCELLED', 'TIMEOUT', 'FAILED', 'COMPLETED'])
def test_only_unfinished_terminal_seeds_resume(state):
    assert handoff.decision(state, False) == 'resume'
    assert handoff.decision(state, True) == 'reuse_complete'


def test_config_allows_only_output_relocation():
    original = dict(runtime=dict(output='/old'), optimization=dict(lr=1e-5), seed=42)
    desired = copy.deepcopy(original)
    desired['runtime']['output'] = '/new'
    handoff.validate_resume_config(original, desired)
    assert original['runtime']['output'] == '/old'
    desired['optimization']['lr'] = 1e-4
    with pytest.raises(ValueError, match='configuration'):
        handoff.validate_resume_config(original, desired)


def test_sacct_expands_cancelled_array_tasks(monkeypatch):
    query = Mock(return_value='123_0|RUNNING|\n123_1|CANCELLED by 77|\n')
    monkeypatch.setattr(handoff.subprocess, 'check_output', query)
    assert handoff.job_states(['123_0', '123_1']) == {'123_0': 'RUNNING', '123_1': 'CANCELLED'}
    assert '--array' in query.call_args.args[0]


def test_submit_only_remaining_seeds_and_gate_on_all_running_jobs(tmp_path, monkeypatch, capsys):
    source = tmp_path / 'source/configs/cluster'
    source.mkdir(parents=True)
    shutil.copyfile(handoff.REPO / 'configs/cluster/bigpurple_a100.yaml', source / 'bigpurple_a100.yaml')
    folder = tmp_path / 'arm'
    folder.mkdir()
    entries = [dict(seed=42, handoff_action='reuse_running', job='90_0'),
               dict(seed=696, handoff_action='reuse_complete', job='90_1'),
               dict(seed=1001, handoff_action='resume', source_job='90_2', job=None)]
    handoff.write(folder / 'entries.json', entries)
    handoff.write(tmp_path / 'manifest.json', dict(aliases=['arm'], downstream_jobs=[], tusz_owner='oldowner'))
    monkeypatch.setattr(handoff, 'job_states', lambda ids: {'90_2': 'TIMEOUT'})
    submit = Mock(side_effect=['100\n', '101\n'])
    mutate = Mock()
    monkeypatch.setattr(handoff.subprocess, 'check_output', submit)
    monkeypatch.setattr(handoff.subprocess, 'run', mutate)
    handoff.submit(tmp_path)
    gpu, cpu = [call.args[0] for call in submit.call_args_list]
    assert '--array=2%5' in gpu and '--hold' in gpu
    assert '--cpus-per-task=2' in gpu and '--time=04:00:00' in gpu
    assert '--partition=a100_dev,a100_short,a100_long' in gpu
    assert '--dependency=afterany:100:90_0' in cpu
    assert not any('gpu' in arg for arg in cpu)
    assert mutate.call_args.args[0] == ['scontrol', 'release', '100']
    assert 'python ' in capsys.readouterr().out
    saved = json.loads((folder / 'entries.json').read_text())
    assert [e['job'] for e in saved] == ['90_0', '90_1', '100_2']
    with pytest.raises(ValueError, match='already submitted'):
        handoff.submit(tmp_path)


def test_wrong_account_cannot_modify_tusz(tmp_path, monkeypatch):
    path = tmp_path / 'manifest.json'
    handoff.write(path, dict(tusz_owner_uid=1, tusz_owner='original'))
    monkeypatch.setattr(handoff.os, 'getuid', lambda: 2, raising=False)
    mutate = Mock()
    monkeypatch.setattr(handoff.subprocess, 'run', mutate)
    with pytest.raises(PermissionError, match='original'):
        handoff.attach_tusz(path)
    mutate.assert_not_called()


def test_missing_seed_blocks_every_publication(tmp_path, monkeypatch):
    aliases = ['geometry', 'static', 'scalar', 'dimension']
    handoff.write(tmp_path / 'manifest.json', dict(aliases=aliases))
    for alias in aliases:
        folder = tmp_path / alias
        folder.mkdir()
        entries = [dict(seed=seed, job='123', output=str(folder / str(seed))) for seed in handoff.SEEDS]
        handoff.write(folder / 'entries.json', entries)
    publish = Mock()
    monkeypatch.setattr(publisher, 'publish', publish)
    assert audit.audit_and_publish(tmp_path) == 1
    state = json.loads((tmp_path / 'completion.json').read_text())
    assert state['status'] == 'incomplete' and len(state['missing']) == 20
    publish.assert_not_called()


def test_tusz_attachment_preserves_dependency_and_full_parallelism(tmp_path, monkeypatch):
    path = tmp_path / 'manifest.json'
    handoff.write(path, dict(tusz_owner_uid=1, tusz_owner='original', completion_job='100',
                            tusz_jobs=['200', '201'], old_controller='99', source_campaign=str(tmp_path)))
    monkeypatch.setattr(handoff.os, 'getuid', lambda: 1, raising=False)
    monkeypatch.setattr(handoff.subprocess, 'check_output', lambda *a, **kw: 'JobState=PENDING')
    mutate = Mock(return_value=Mock(returncode=0, stdout='JobState=PENDING'))
    monkeypatch.setattr(handoff.subprocess, 'run', mutate)
    handoff.attach_tusz(path)
    commands = [c.args[0] for c in mutate.call_args_list]
    assert ['scontrol', 'update', 'JobId=200', 'Dependency=afterok:100', 'ArrayTaskThrottle=5'] in commands
    assert ['scontrol', 'update', 'JobId=201', 'Dependency=afterok:100', 'ArrayTaskThrottle=5'] in commands
    assert commands[-1] == ['scancel', '99']


def test_repeated_launch_cannot_duplicate_jobs(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'fcntl', SimpleNamespace(LOCK_EX=2, flock=lambda *a: None))
    prepare = Mock(return_value=tmp_path / 'experiment')
    submit = Mock()
    monkeypatch.setattr(handoff, 'prepare', prepare)
    monkeypatch.setattr(handoff, 'submit', submit)
    handoff.launch_once(tmp_path / 'source', tmp_path / 'destination')
    with pytest.raises(ValueError, match='already'):
        handoff.launch_once(tmp_path / 'source', tmp_path / 'destination')
    prepare.assert_called_once()
    submit.assert_called_once()


@pytest.mark.parametrize('corrupt_selection', [False, True])
def test_complete_audit_requires_validation_selected_weights(tmp_path, monkeypatch, corrupt_selection):
    import torch
    import yaml
    monkeypatch.setitem(sys.modules, 'fcntl', SimpleNamespace(LOCK_EX=2, flock=lambda *a: None))
    aliases = ['geometry', 'static', 'scalar', 'dimension']
    root = tmp_path / 'results'
    monkeypatch.setattr(publisher, 'RESULTS_ROOT', root)
    handoff.write(tmp_path / 'manifest.json', dict(aliases=aliases, publication_root=str(root),
                                                created_at_new_york='260920-002000'))
    for alias in aliases:
        folder = tmp_path / alias
        folder.mkdir()
        entries = []
        for seed in sorted(handoff.SEEDS):
            output = folder / str(seed)
            output.mkdir()
            config = dict(seed=seed, runtime=dict(output=str(output)))
            cfg = output / 'config.yaml'
            cfg.write_text(yaml.safe_dump(config))
            selection = dict(epoch=19, score=0.83)
            weights = dict(weight=torch.tensor([1.]))
            torch.save(dict(epoch=20, config=config, model=weights,
                            extra=dict(best=dict(balanced_accuracy=selection))), output / 'last.pth')
            torch.save(weights, output / 'best-balanced_accuracy.pth')
            (output / 'validation.jsonl').write_text(json.dumps(dict(epoch=19, balanced_accuracy=0.83)))
            payload = dict(balanced_accuracy=dict(selection=selection.copy(),
                           test=dict(balanced_accuracy=0.8, auroc=0.9, auprc=0.9)))
            if corrupt_selection and alias == 'static' and seed == 42:
                payload['balanced_accuracy']['selection']['score'] = 0.9
            handoff.write(output / 'result.json', payload)
            entries.append(dict(seed=seed, job='123', config=str(cfg), output=str(output), dataset='tuab'))
        handoff.write(folder / 'entries.json', entries)
    assert audit.audit_and_publish(tmp_path) == int(corrupt_selection)
    if corrupt_selection:
        assert not root.exists()
        status = json.loads((tmp_path / 'completion.json').read_text())
        assert len(status['missing']) == 1
    else:
        assert len(list(root.glob('*/seed_results.csv'))) == 4
        assert (root / 'RESULTS.md').exists()
