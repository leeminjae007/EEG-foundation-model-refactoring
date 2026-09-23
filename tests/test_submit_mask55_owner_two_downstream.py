from scripts import submit_mask55_owner_two_downstream as campaign


def test_owner_ablation_scope_and_final_defaults():
    assert campaign.ARMS == ('ours-lite', 'enc-average-3s')
    assert len(campaign.DATASETS) == 10
    assert 'tuab' not in campaign.DATASETS
    assert 'tusl' not in campaign.DATASETS
    assert campaign.HP['mentalarithmetic'] == {
        'learning_rate': 1e-4, 'weight_decay': .02, 'head_dropout': .1,
    }
    assert campaign.HP['physionet_mi'] == campaign.HP['mentalarithmetic']
    assert campaign.common.resources('gl40s', 'chb')['partitions'] == 'gl40s_short,gl40s_long'
