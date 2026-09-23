from scripts import finalize_experiment
from scripts import submit_mask55_tuab_comparison as campaign


def test_tuab_comparison_assignments_and_hp():
    assert len(campaign.ARMS['hk4935']) == 3
    assert len(campaign.ARMS['yc8820']) == 5
    assert len(campaign.ARMS['ml10266']) == 2
    assert campaign.ARMS['ml10266']['enc-average-3s'][1] == 'gl40s'
    assert all(gpu == 'a100' for _, gpu in list(campaign.ARMS['hk4935'].values()))
    assert campaign.checked_hp(5e-4, .05, .3)['early_stopping']['patience'] == 10


def test_finalizer_accepts_only_verified_early_stopping():
    config = {'optimization': {'epochs': 20, 'early_stopping': dict(campaign.EARLY_STOPPING)}}
    saved = {'config': config, 'epoch': 11,
             'extra': {'partial_epoch_smoke': False, 'early_stopped': True}}
    payload = {'_training': {'epochs_completed': 11, 'early_stopped': True}}
    assert finalize_experiment.training_complete(saved, config, payload)
    saved['extra']['early_stopped'] = False
    assert not finalize_experiment.training_complete(saved, config, payload)
    saved['extra']['early_stopped'] = True
    payload['_training']['epochs_completed'] = 10
    assert not finalize_experiment.training_complete(saved, config, payload)
