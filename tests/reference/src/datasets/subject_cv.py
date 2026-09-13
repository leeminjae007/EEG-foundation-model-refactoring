"""Opt-in subject-disjoint views; never rewrite the benchmark LMDB splits."""

import re

from torch.utils.data import Dataset


def physical_subject(dataset_name, key):
    patterns = {
        'stress': r'^(Subject\d+)_([12])-\d+$',
        'bciciv2a': r'^(A\d{2})[ET]-',
    }
    if dataset_name not in patterns:
        raise ValueError('Subject CV is approved only for stress and bciciv2a')
    match = re.match(patterns[dataset_name], key)
    if match is None:
        raise ValueError(f'Unrecognized {dataset_name} subject key: {key}')
    return match.group(1)


class SubjectSubset(Dataset):
    """Subset with the geometry metadata required by the unchanged trainer."""

    def __init__(self, dataset, indices, dataset_name):
        self.dataset = dataset
        self.indices = tuple(indices)
        self.keys = tuple(dataset.keys[i] for i in self.indices)
        self.subject_ids = tuple(physical_subject(dataset_name, k) for k in self.keys)
        self.recording_ids = tuple(dataset.recording_ids[i] for i in self.indices)
        for name in ('channel_names', 'channel_validity', 'channel_coordinates',
                     'channel_region_ids'):
            setattr(self, name, getattr(dataset, name))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        item = dict(self.dataset[self.indices[index]])
        item['subject_id'] = self.subject_ids[index]
        return item


def subject_inventory(datasets, dataset_name):
    groups = {
        split: sorted({physical_subject(dataset_name, key) for key in ds.keys})
        for split, ds in datasets.items()
    }
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if set(groups[a]) & set(groups[b]):
            raise ValueError(f'Original {a}/{b} subjects overlap')
    return groups


def apply_subject_cv(datasets, dataset_name, policy):
    if policy is None:
        return datasets
    if policy.get('method') != 'loso_train_only':
        raise ValueError('Only train-only LOSO is supported')
    groups = subject_inventory(datasets, dataset_name)
    held_out = policy['held_out_subject']
    if held_out not in groups['train']:
        raise ValueError('LOSO held-out subject must belong to original training')
    original = datasets['train']
    subjects = [physical_subject(dataset_name, key) for key in original.keys]
    train = [i for i, subject in enumerate(subjects) if subject != held_out]
    val = [i for i, subject in enumerate(subjects) if subject == held_out]
    if not train or not val:
        raise ValueError('Empty LOSO training or validation fold')
    views = dict(datasets)
    views['train'] = SubjectSubset(original, train, dataset_name)
    views['val'] = SubjectSubset(original, val, dataset_name)
    # Remap test metadata only; the fold run is forbidden from evaluating it.
    views['test'] = SubjectSubset(datasets['test'], range(len(datasets['test'])), dataset_name)
    return views
