"""Geometry 이식의 원본 동등성, 중복 제거, 초기화, 학습 재개를 검사한다."""
import copy
import importlib.util
import json
import os
import subprocess
import sys

import pytest
import torch
import yaml

from scripts.convert_checkpoint import convert
from src.model import PretrainModel
from src.modules.masking import make_masks, gather_targets
from src.modules.loss import reconstruction_loss
from src.training.runtime import ROOT, set_paths
from src.training.checkpoint import load_checkpoint

set_paths()
torch.set_num_threads(2)


def config():
    return yaml.safe_load((ROOT / 'configs/pretrain.yaml').read_text())


def test_mask_matches_original_and_partitions_grid():
    spec = importlib.util.spec_from_file_location('geometry_oracle_masks', ROOT / 'tests/reference/src/utils/masking.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    model = PretrainModel(config(), torch.device('cpu'))
    coordinates = model.backbone.default_channel_coordinates
    settings = {'policy': 'geometry_tubelet'}
    # 세 실험 모두에서 원본 numpy sampling 순서와 torch RNG 소비를 보존한다.
    for radius, duration, ratio in [(35, 2, .5), (50, 5, .5), (50, 5, .55)]:
        settings.update(min_radius_degrees=radius, max_radius_degrees=radius + 40,
                        min_time_patches=duration, max_time_patches=10 if duration == 2 else 15,
                        mask_ratio=ratio)
        for seed in [42, 1234, 696, 1001, 3407]:
            torch.manual_seed(seed)
            masks = make_masks(3, 19, 30, settings, torch.device('cpu'), coordinates)
            after = torch.get_rng_state()
            torch.manual_seed(seed)
            expected = original.build_masking_policy(settings)(
                3, 19, 30, torch.ones(3, 19, 30, dtype=torch.bool), channel_coordinates=coordinates)
            assert torch.equal(after, torch.get_rng_state())
            for key, value in masks.items():
                if key == 'masking_diagnostics':
                    for name, metric in value.items():
                        assert torch.equal(metric, expected[key][name])
                else:
                    assert torch.equal(value, expected[key]), key
            target, context = masks['target_mask'], masks['context_mask']
            assert not (target & context).any()
            assert (target | context).all()
            assert (target.sum((1, 2)) == round(570 * ratio)).all()
            assert masks['target_blocks'].shape == (3, 1, 19, 30)
            indices = torch.arange(570).reshape(1, 19, 30, 1).expand(3, -1, -1, -1)
            gathered = gather_targets(indices, masks['target_blocks'])
            assert all(len(row.unique()) == round(570 * ratio) for row in gathered)


def test_default_changes_only_masking_and_preserves_initialization():
    new = config()
    old = yaml.safe_load((ROOT / 'configs/gr9_1.yaml').read_text())
    restored = copy.deepcopy(new)
    restored['masking'] = old['masking']
    restored['runtime']['output'] = old['runtime']['output']
    assert restored == old
    assert new['masking']['mask_ratio'] == .5
    torch.manual_seed(918)
    a = PretrainModel(new, torch.device('cpu'))
    rng_a = torch.get_rng_state()
    torch.manual_seed(918)
    b = PretrainModel(old, torch.device('cpu'))
    assert torch.equal(rng_a, torch.get_rng_state())
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key]), key


def test_geometry_forward_backward_matches_frozen_original():
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    checkpoint = torch.load(ROOT / 'tests/reference/checkpoint-epoch-0040.pth', map_location='cpu')
    state, mapping = convert(checkpoint)
    model = PretrainModel(config(), torch.device('cpu'))
    model.load_state_dict(state, strict=True)
    torch.manual_seed(47)
    masks = make_masks(2, 19, 30, config()['masking'], torch.device('cpu'),
                       model.mask_coordinates)
    torch.save(masks, ROOT / 'outputs/geometry_reve_masks.pth')
    subprocess.run([sys.executable, str(ROOT / 'tests/export_geometry_reference.py')],
                   cwd=ROOT, env=env, check=True)
    reference = torch.load(ROOT / 'outputs/geometry_reve_oracle.pth', map_location='cpu')
    torch.set_rng_state(reference['rng_before'])
    model.train()
    prediction, _ = model(reference['signals'], masks)
    target = gather_targets(reference['signals'].unfold(-1, 200, 200), masks['target_blocks'])
    assert prediction.shape == (2, 285, 200)
    assert torch.equal(target, reference['target'])
    loss = reconstruction_loss(prediction, target, masks['target_token_valid'], .1)
    loss.backward()
    assert torch.equal(torch.get_rng_state(), reference['rng_after'])
    torch.testing.assert_close(prediction, reference['prediction'], rtol=0, atol=0)
    torch.testing.assert_close(loss, reference['loss'], rtol=0, atol=0)
    gradients = dict(model.named_parameters())
    maximum = 0.
    for old, expected in reference['gradients'].items():
        actual = gradients[mapping[old]].grad
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        maximum = max(maximum, float((actual - expected).abs().max()))
    (ROOT / 'outputs/geometry_reve_equivalence.json').write_text(json.dumps({
        'prediction_max_abs': float((prediction - reference['prediction']).abs().max()),
        'loss_max_abs': float((loss - reference['loss']).abs()),
        'gradient_max_abs': maximum, 'gradient_tensors': len(reference['gradients']),
        'rng_equal': True, 'target_tokens': 285, 'context_tokens': 285,
        'device': 'cpu', 'dtype': 'float32', 'batch': 2}, indent=2) + '\n')


def test_geometry_requires_coordinates_and_rejects_short_grid():
    settings = config()['masking']
    with pytest.raises(ValueError, match='requires channel coordinates'):
        make_masks(1, 19, 30, settings, torch.device('cpu'))
    with pytest.raises(ValueError, match='minimum tubelet time span'):
        make_masks(1, 19, 1, settings, torch.device('cpu'), torch.ones(19, 3))
    with pytest.raises(ValueError, match='unsupported masking policy'):
        make_masks(1, 19, 30, {'policy': 'typo'}, torch.device('cpu'))


def test_resume_rejects_different_mask_before_loading_weights(tmp_path):
    path = tmp_path / 'old.pth'
    torch.save({'config': {'masking': {'policy': 'ijepa_multiblock'}}}, path)
    with pytest.raises(ValueError, match='masking config differs'):
        load_checkpoint(path, None, None, None, torch.device('cpu'), 0,
                        expected_masking=config()['masking'])


def test_reve_radius_uses_physical_units_and_285_unique_targets():
    import numpy as np
    import mne
    from scipy.spatial import KDTree
    from src.modules.geometry_masking import physical_channel_coordinates, GeometryTubeletMaskingPolicy
    from src.data.electrode_geometry import canonicalize_channel_name
    cfg = config()
    settings = cfg['masking']
    assert settings['radius_m'] == .03
    assert settings['distance_metric'] == 'euclidean_m'
    assert (settings['min_time_patches'], settings['max_time_patches']) == (2, 15)
    positions = physical_channel_coordinates(cfg['data']['channel_names'])
    montage = mne.channels.make_standard_montage('standard_1020').get_positions()['ch_pos']
    lookup = {name.upper(): value for name, value in montage.items()}
    expected = np.stack([lookup[canonicalize_channel_name(name)] for name in cfg['data']['channel_names']])
    np.testing.assert_allclose(positions.numpy(), expected, rtol=1e-6)
    # 모델 PE 좌표의 정규화 크기를 물리 단위로 잘못 쓰지 않는지 확인한다.
    preserved = GeometryTubeletMaskingPolicy._coordinates(positions, 1, 19, normalize=False)[0]
    assert torch.equal(preserved, positions)
    distances = torch.cdist(positions, positions).numpy()
    tree = KDTree(expected)
    neighbors = []
    for i in range(19):
        actual = np.flatnonzero(distances[i] <= .03).tolist()
        assert actual == sorted(tree.query_ball_point(expected[i], .03))
        neighbors.append(len(actual))
    for seed in [42, 1234, 696, 1001, 3407]:
        torch.manual_seed(seed)
        masks = make_masks(4, 19, 30, settings, torch.device('cpu'), positions)
        assert (masks['target_mask'].sum((1, 2)) == 285).all()
        assert (masks['context_mask'].sum((1, 2)) == 285).all()
        assert not (masks['target_mask'] & masks['context_mask']).any()
        assert masks['target_blocks'].shape == (4, 1, 19, 30)
        radius = masks['masking_diagnostics']['geometry_mean_radius_m']
        assert abs(float(radius) - .03) < 1e-7
    (ROOT / 'outputs/reve_radius_verification.json').write_text(json.dumps({
        'radius_m': .03, 'metric': 'euclidean', 'channel_names': cfg['data']['channel_names'],
        'positions_m': positions.tolist(), 'neighbor_counts_including_self': neighbors,
        'matches_reve_kdtree_neighbors': True, 'target_tokens': 285, 'context_tokens': 285,
        'time_patches_inclusive': [2, 15]}, indent=2) + '\n')
