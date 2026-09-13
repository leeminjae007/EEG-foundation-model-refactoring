"""Minimal experiment provenance for the current EEG MAE pipeline."""

import hashlib
from importlib import metadata
import json
import os
import platform
import sys
import tempfile
from pathlib import Path


PACKAGES = (
    'torch',
    'numpy',
    'lmdb',
    'PyYAML',
    'mne',
    'scipy',
    'scikit-learn',
    'wandb',
)

SOURCE_SUFFIXES = (
    '.py', '.yaml', '.sbatch', '.md', '.txt', '.json', '.jsonl')

SOURCE_DIRECTORIES = ('configs', 'scripts', 'src', 'tests')


def canonical_sha256(value):
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(',', ':'),
        default=str,
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _source_files(repository):
    """Return only files that define the executable experiment source.

    Result and log trees are intentionally never traversed.  Filtering them
    after ``Path.rglob`` still walks every W&B artifact/checkpoint and can race
    with normal cleanup, even though those files are not part of the hash.
    """
    repository = Path(repository)
    files = [
        path for path in repository.iterdir()
        if path.is_file() and path.suffix in SOURCE_SUFFIXES
    ]
    for directory_name in SOURCE_DIRECTORIES:
        directory = repository / directory_name
        if not directory.is_dir():
            continue
        files.extend(
            path for path in directory.rglob('*')
            if path.is_file()
            and path.suffix in SOURCE_SUFFIXES
            and '__pycache__' not in path.parts
        )
    return sorted(set(files))


def source_tree_sha256(repository=None):
    repository = (
        Path(__file__).resolve().parents[2]
        if repository is None else Path(repository)
    )
    hasher = hashlib.sha256()
    for path in _source_files(repository):
        relative = str(path.relative_to(repository)).encode()
        content = path.read_bytes()
        hasher.update(len(relative).to_bytes(8, 'big'))
        hasher.update(relative)
        hasher.update(len(content).to_bytes(8, 'big'))
        hasher.update(content)
    return hasher.hexdigest()


def build_reproducibility_manifest(config, dataset_fingerprint=None):
    return {
        'config_sha256': canonical_sha256(config),
        'source_tree_sha256': source_tree_sha256(),
        'dataset': dataset_fingerprint,
        'packages': {
            package: metadata.version(package)
            for package in PACKAGES
        },
        'platform': platform.platform(),
        'python': sys.version,
    }


def write_reproducibility_manifest(manifest, output_path):
    output_path = Path(output_path)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix='.reproducibility-',
        suffix='.tmp',
        dir=str(output_path.parent),
        text=True,
    )
    with os.fdopen(descriptor, 'w', encoding='utf-8') as output:
        json.dump(manifest, output, indent=2, sort_keys=True)
        output.write('\n')
    os.replace(temporary_path, output_path)
