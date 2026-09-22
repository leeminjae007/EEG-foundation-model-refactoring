"""Distributed loader for the processed TUEG LMDB."""

import hashlib
import pickle
import random

import lmdb
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

class PretrainingDataset(Dataset):
    def __init__(self, dataset_dir, channels, num_patches, patch_samples):
        self.dataset_dir = str(dataset_dir)
        self.sample_shape = (
            int(channels),
            int(num_patches),
            int(patch_samples),
        )
        self.database = lmdb.open(
            self.dataset_dir,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
        )
        with self.database.begin(write=False) as transaction:
            self.keys = pickle.loads(transaction.get(b'__keys__'))
            entries = transaction.stat()['entries']
        database_info = self.database.info()
        self.database.close()
        self.database = None

        key_hasher = hashlib.sha256()
        for key in self.keys:
            encoded = key.encode()
            key_hasher.update(len(encoded).to_bytes(8, 'big'))
            key_hasher.update(encoded)
        self.fingerprint = {
            'num_samples': len(self.keys),
            'lmdb_entries': entries,
            'ordered_keys_sha256': key_hasher.hexdigest(),
            'last_txnid': database_info['last_txnid'],
            'last_pgno': database_info['last_pgno'],
        }

    def _database(self):
        if self.database is None:
            self.database = lmdb.open(
                self.dataset_dir,
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
            )
        return self.database

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        with self._database().begin(write=False) as transaction:
            patch = pickle.loads(
                transaction.get(self.keys[index].encode()))
        assert tuple(patch.shape) == self.sample_shape
        signal = torch.from_numpy(patch).float().reshape(
            self.sample_shape[0], -1)
        return signal, index

    def close(self):
        if self.database is not None:
            self.database.close()
            self.database = None

    def __del__(self):
        self.close()


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_pretraining_loader(
    dataset_dir,
    batch_size,
    pin_mem,
    num_workers,
    world_size,
    rank,
    seed,
    drop_last,
    channels,
    num_patches,
    patch_samples,
):
    dataset = PretrainingDataset(
        dataset_dir,
        channels=channels,
        num_patches=num_patches,
        patch_samples=patch_samples,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
        drop_last=drop_last,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return dataset, loader, sampler
