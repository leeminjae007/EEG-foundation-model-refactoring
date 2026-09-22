from scripts import retry_mask55_hk_siena_tusz_a100 as retry


def test_retryable_indices_uses_only_terminal_missing_entries(monkeypatch):
    entries = [
        dict(index=0, dataset='siena'),
        dict(index=1, dataset='siena'),
        dict(index=2, dataset='siena'),
        dict(index=3, dataset='siena'),
        dict(index=180, dataset='tusz'),
    ]
    monkeypatch.setattr(retry, 'complete', lambda entry: entry['index'] == 1)
    states = {0: 'FAILED', 1: 'COMPLETED', 2: 'RUNNING', 3: 'TIMEOUT'}
    assert retry.retryable_indices(entries, 'siena', states, {3}) == [0]


def test_pending_original_excludes_bad_nodes(monkeypatch):
    calls = []

    def fake_run(command, check):
        calls.append(command)

    monkeypatch.setattr(retry.subprocess, 'run', fake_run)
    monkeypatch.setattr(retry.subprocess, 'check_output',
                        lambda command, text: 'JobId=27681546_2 ExcNodeList=a100-4011,a100-4024')
    retry.exclude_pending_original('27681546', {2: 'PENDING', 3: 'RUNNING'},
                                   ['a100-4011', 'a100-4024'])
    assert calls == [['scontrol', 'update', 'JobId=27681546',
                      'ExcNodeList=a100-4011,a100-4024']]


def test_probe_and_retry_wait_for_cuda(tmp_path, monkeypatch):
    calls = []

    def fake_submit(command, text):
        calls.append(command)
        return str(9000 + len(calls))

    monkeypatch.setattr(retry.subprocess, 'check_output', fake_submit)
    logs = tmp_path / 'logs'
    probe = retry.submit_probe(logs, ['a100-4011'], 'system')
    job = retry.submit_retry(tmp_path, 'siena', [4, 6], probe, ['a100-4011'],
                             dict(partitions='a100_short,a100_long',
                                  time_limits=dict(siena='04:00:00')), 'system')
    assert (probe, job) == ('9001', '9002')
    assert '--exclude=a100-4011' in calls[0]
    assert 'CUDA_PROBE_PASSED' in calls[0][-1]
    assert '--dependency=afterok:9001' in calls[1]
    assert '--array=4,6%5' in calls[1]
    assert '--exclude=a100-4011' in calls[1]
