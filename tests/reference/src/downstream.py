"""Single-GPU downstream fine-tuning for the retained EEG MAE."""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import re

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler
import yaml

from src.datasets.registry import get_dataset_spec
from src.utils.checkpointing import atomic_save, capture_rng_state, restore_rng_state
from src.models.finetune.scalp_backbone import build_raw_eeg_encoder
from src.models.finetune.task_model import TaskModel
from src.tracking import init_wandb
from src.utils.metrics import downstream_metrics
from src.utils.classification_losses import classification_loss
from src.utils.downstream_diagnostics import (
    binary_calibration,
    classification_loss_diagnostics,
    confusion_diagnostics,
    linear_cka,
    nearest_centroid_probe,
    regression_calibration,
    subject_metric_diagnostics,
)
from src.utils.schedulers import GroupCosineScheduler
from src.utils.model_diagnostics import (
    set_model_diagnostics, collect_model_diagnostics, gate_diagnostics,
    initialization_fingerprints, compare_initialization, write_initialization_audit,
)
from src.utils.reproducibility import (
    build_reproducibility_manifest,
    canonical_sha256,
    write_reproducibility_manifest,
)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


class ExactDistributedSampler(Sampler):
    """Shard evaluation data across ranks without padding or duplication."""

    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.world_size - 1) // self.world_size)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_datasets(
    dataset_dir,
    dataset_name,
    sample_scale_correction=None,
):
    dataset_class = get_dataset_spec(dataset_name).dataset_class
    datasets = {
        split: dataset_class(dataset_dir, split)
        for split in ('train', 'val', 'test')
    }
    empty = [split for split, dataset in datasets.items() if len(dataset) == 0]
    if empty:
        raise ValueError(
            f'{dataset_name} has empty processed splits at {dataset_dir}: '
            f'{empty}'
        )
    if sample_scale_correction is not None:
        for dataset in datasets.values():
            dataset.sample_scale_correction = dict(sample_scale_correction)
    for dataset in datasets.values():
        dataset.enable_coordinate_only_channels()
    return datasets


def _diagnostic_subject_ids(dataset):
    values = [str(s) for s in getattr(dataset, 'subject_ids', ())]
    if type(dataset).__name__ == 'StressDataset':
        values = [re.sub(r'^(Subject\d+)_[12]$', r'\1', s) for s in values]
    return [s for s in values if s.lower() not in {'unavailable', 'unknown', 'none', ''}]


def _make_loaders(
    datasets,
    batch_size,
    num_workers,
    rank,
    world_size,
    seed=42,
):
    train_sampler = DistributedSampler(
        datasets['train'],
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(seed),
    )
    loaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    }
    for split in ('val', 'test'):
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            sampler=ExactDistributedSampler(
                datasets[split], rank, world_size),
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders


def _build_eeg_mae_model(config, spec, checkpoint=None):
    checkpoint_path = config['model']['checkpoint']
    if checkpoint is None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    pretrain_config = checkpoint['resolved_config']
    assert (
        pretrain_config['experiment']['trial_name']
        == config['experiment']['trial_name']
    )
    encoder = build_raw_eeg_encoder(pretrain_config)
    transfer_mode = config['model'].get('transfer_mode', 'full')
    if transfer_mode != 'random':
        encoder.load_state_dict(checkpoint['context_encoder'], strict=True)
    elif pretrain_config['encoder'].get('weight_initialization') == 'kaiming_normal_fan_out_relu':
        from src.models.factory import initialize_cbramod_weights
        # Match the baseline's random initialization distribution, without
        # shifting the downstream head's RNG stream versus full/frozen controls.
        with torch.random.fork_rng(devices=[]):
            encoder.apply(initialize_cbramod_weights)
    embed_dim = pretrain_config['encoder']['embed_dim']
    patch_samples = pretrain_config['patch_encoder']['patch_samples']
    if pretrain_config['latent_tokenizer'].get(
        'mode', 'fixed_region'
    ) == 'none':
        spatial_tokens = spec.num_channels
    else:
        spatial_tokens = pretrain_config['latent_tokenizer']['num_latents']
    assert spec.signal_length % patch_samples == 0
    if config['data']['dataset'] == 'isruc':
        if transfer_mode != 'full':
            raise ValueError('ISRUC transfer controls are not implemented')
        from src.models.finetune.model_for_isruc import Model
        return Model(
            encoder=encoder,
            embed_dim=embed_dim,
            num_latents=spatial_tokens,
            num_patches=spec.signal_length // patch_samples,
            dropout=config['model']['head_dropout'],
            pooling=config['model']['pooling'],
            head_activation=config['model'].get(
                'head_activation', 'gelu'),
        )
    return TaskModel(
        encoder,
        embed_dim,
        spec.num_outputs,
        head_dropout=config['model']['head_dropout'],
        pooling=config['model']['pooling'],
        num_latents=spatial_tokens,
        num_patches=spec.signal_length // patch_samples,
        head_hidden_tokens=config['model'].get('head_hidden_tokens'),
        head_activation=config['model'].get('head_activation', 'gelu'),
        transfer_mode=transfer_mode,
    )


def build_downstream_model(config):
    spec = get_dataset_spec(config['data']['dataset'])
    return _build_eeg_mae_model(config, spec)


def _forward(model, batch, device):
    return model(
        batch['x'].to(device, non_blocking=True),
        channel_coordinates=batch['channel_coordinates'].to(
            device, non_blocking=True),
        channel_region_ids=batch['channel_region_ids'].to(
            device, non_blocking=True),
        channel_validity=batch['channel_validity'].to(
            device, non_blocking=True),
    )


def _loss(task, logits, labels, optimization):
    if task in {'binary', 'multiclass'}:
        return classification_loss(task, logits, labels, optimization)
    predictions = logits.reshape(-1)
    targets = labels.to(dtype=logits.dtype).reshape(-1)
    return F.mse_loss(predictions, targets)


def _summary_embedding(model):
    classifier = _classifier_module(model)
    summary = getattr(classifier, 'last_summary_embedding', None)
    if summary is not None:
        return summary
    module = (
        model.module if isinstance(model, DistributedDataParallel) else model)
    return getattr(module, 'last_summary_embedding', None)


def _classifier_module(model):
    module = (
        model.module if isinstance(model, DistributedDataParallel) else model)
    return getattr(module, 'classifier', None)


def _last_forward_diagnostics(model):
    classifier = _classifier_module(model)
    diagnostics = {}
    if classifier is not None:
        diagnostics.update({
            key: float(value)
            for key, value in classifier.last_diagnostics.items()
        })
    diagnostics.update({key: float(value) for key, value in collect_model_diagnostics(model).items()})
    return diagnostics


def _mean_diagnostics(records):
    if not records:
        return {}
    return {
        key: float(np.mean([record[key] for record in records if key in record]))
        for key in sorted({key for record in records for key in record})
    }


def _prediction_diagnostics(
    task,
    logits,
    labels,
):
    diagnostics = {
        'logit_std': float(logits.float().std(unbiased=False)),
    }
    if task == 'regression':
        predictions = logits.float().reshape(-1)
        targets = labels.float().reshape(-1)
        diagnostics['prediction_mean'] = float(predictions.mean())
        diagnostics['target_mean'] = float(targets.mean())
        diagnostics['mean_error'] = float(
            predictions.mean() - targets.mean())
        diagnostics['prediction_std'] = float(
            predictions.std(unbiased=False))
        diagnostics['target_std'] = float(
            targets.std(unbiased=False))
        diagnostics.update(regression_calibration(predictions, targets))
        return diagnostics
    if task == 'binary':
        probabilities = logits.float().sigmoid().reshape(-1)
        targets = labels.reshape(-1).long()
        predictions = probabilities.ge(0.5).long()
        num_classes = 2
        diagnostics['probability_mean'] = float(probabilities.mean())
        diagnostics['probability_std'] = float(
            probabilities.std(unbiased=False))
        diagnostics['true_negative'] = int(
            ((predictions == 0) & (targets == 0)).sum())
        diagnostics['false_positive'] = int(
            ((predictions == 1) & (targets == 0)).sum())
        diagnostics['false_negative'] = int(
            ((predictions == 0) & (targets == 1)).sum())
        diagnostics['true_positive'] = int(
            ((predictions == 1) & (targets == 1)).sum())
        diagnostics.update(binary_calibration(logits, labels))
    else:
        predictions = logits.argmax(dim=-1)
        num_classes = logits.shape[-1]
    predicted = torch.bincount(
        predictions.reshape(-1).cpu(), minlength=num_classes)
    targets = torch.bincount(
        labels.reshape(-1).long().cpu(), minlength=num_classes)
    for index in range(num_classes):
        diagnostics[f'predicted_class_count/{index}'] = int(predicted[index])
        diagnostics[f'target_class_count/{index}'] = int(targets[index])
    diagnostics['predicted_class_count/nonzero'] = int(predicted.gt(0).sum())
    diagnostics.update(confusion_diagnostics(logits, labels))
    return diagnostics


@torch.no_grad()
def evaluate(
    model,
    loader,
    task,
    device,
):
    model.eval()
    module = (
        model.module if isinstance(model, DistributedDataParallel) else model)
    classifier = _classifier_module(model)
    previous_diagnostics = (
        classifier.diagnostics_enabled if classifier is not None else False)
    if classifier is not None:
        classifier.diagnostics_enabled = True
    logits = []
    labels = []
    forward_diagnostics = []
    subject_ids = []
    fixed_probe_embeddings = None
    probe_samples = {}
    for batch_index, batch in enumerate(loader):
        set_model_diagnostics(module, batch_index == 0)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            prediction = _forward(module, batch, device).detach()
        summary_embeddings = _summary_embedding(model).detach().cpu()
        if fixed_probe_embeddings is None:
            fixed_probe_embeddings = summary_embeddings
        forward_diagnostics.append(_last_forward_diagnostics(model))
        truth = batch['label'].to(device, non_blocking=True)
        if truth.ndim > 1:
            repeats = truth.reshape(truth.shape[0], -1).shape[1]
            batch_subject_ids = [
                subject_id
                for subject_id in batch['subject_id']
                for _ in range(repeats)
            ]
            subject_ids.extend(batch_subject_ids)
            prediction = prediction.reshape(-1, prediction.shape[-1])
            truth = truth.reshape(-1)
        else:
            batch_subject_ids = list(batch['subject_id'])
            subject_ids.extend(batch_subject_ids)
        # Correct recording-vs-person identity in diagnostics only. Splits,
        # sample order and published loader behavior remain unchanged.
        if type(loader.dataset).__name__ == 'StressDataset':
            batch_subject_ids = [re.sub(r'^(Subject\d+)_[12]$', r'\1', str(s))
                                 for s in batch_subject_ids]
            subject_ids[-len(batch_subject_ids):] = batch_subject_ids
        # Align one diagnostic embedding with each downstream label.  Standard
        # heads expose [B,D], while ISRUC may expose [B,20,D] (or a denser
        # per-epoch tensor).  The label count is therefore the stable contract;
        # every remaining axis belongs to that label's representation.
        label_count = truth.numel()
        if summary_embeddings.numel() % label_count != 0:
            raise ValueError(
                'summary embeddings must align with downstream labels: '
                f'embeddings={tuple(summary_embeddings.shape)} '
                f'labels={tuple(truth.shape)}')
        flattened_embeddings = summary_embeddings.reshape(label_count, -1)
        for embedding, label, subject_id in zip(
            flattened_embeddings,
            truth.detach().cpu(),
            batch_subject_ids,
        ):
            # Bound each person-class cell rather than taking only a person's
            # first (often all-negative) windows. Diagnostic sampling is deterministic.
            key = (str(subject_id), int(label) if task != 'regression' else 'continuous')
            samples = probe_samples.setdefault(key, [])
            if len(samples) < 16:
                samples.append((embedding, float(label)))
        logits.append(prediction)
        labels.append(truth)
    if classifier is not None:
        classifier.diagnostics_enabled = previous_diagnostics
    set_model_diagnostics(module, False)
    logits = torch.cat(logits)
    labels = torch.cat(labels)
    subject_diagnostics = subject_metric_diagnostics(
        task, logits, labels, subject_ids
    )
    sampled = [
        (subject_key[0], embedding, label)
        for subject_key, samples in sorted(probe_samples.items())
        for embedding, label in samples
    ]
    probe_diagnostics = {}
    if sampled:
        probe_embeddings = torch.stack([item[1] for item in sampled])
        probe_subjects = [item[0] for item in sampled]
        probe_labels = torch.tensor([item[2] for item in sampled])
        if task == 'regression':
            boundaries = torch.quantile(
                probe_labels,
                torch.tensor([0.2, 0.4, 0.6, 0.8]),
            )
            label_categories = torch.bucketize(
                probe_labels, boundaries).tolist()
            label_name = 'target_quantile'
        else:
            label_categories = probe_labels.long().tolist()
            label_name = 'class'
        known_subject_indices = [i for i, s in enumerate(probe_subjects)
                                 if str(s).lower() not in {'unavailable', 'unknown', 'none', ''}]
        subject_probe = (nearest_centroid_probe(
            probe_embeddings[known_subject_indices],
            [probe_subjects[i] for i in known_subject_indices])
            if known_subject_indices else {'num_classes': 0, 'chance_normalized_accuracy': float('nan')})
        label_probe = nearest_centroid_probe(
            probe_embeddings, label_categories)
        probe_diagnostics.update({
            f'probe/subject_{key}': value
            for key, value in subject_probe.items()
            if np.isfinite(value)
        })
        probe_diagnostics.update({
            f'probe/{label_name}_{key}': value
            for key, value in label_probe.items()
            if np.isfinite(value)
        })
        subject_defined = np.isfinite(subject_probe['chance_normalized_accuracy'])
        label_defined = np.isfinite(label_probe['chance_normalized_accuracy'])
        probe_diagnostics['probe/subject_defined'] = int(subject_defined)
        probe_diagnostics[f'probe/{label_name}_defined'] = int(label_defined)
        probe_diagnostics['probe/person_class_stratified'] = int(task != 'regression')
        if subject_defined and label_defined:
            probe_diagnostics['probe/subject_minus_label_normalized'] = (
                subject_probe['chance_normalized_accuracy']
                - label_probe['chance_normalized_accuracy'])
    return (
        downstream_metrics(task, logits, labels),
        {
            **_mean_diagnostics(forward_diagnostics),
            **_prediction_diagnostics(
                task,
                logits,
                labels,
            ),
            **{
                f'subject_macro/{key}': value
                for key, value in subject_diagnostics['macro'].items()
            },
            **probe_diagnostics,
        },
        subject_diagnostics,
        fixed_probe_embeddings,
    )


@torch.no_grad()
def _fixed_probe_embeddings(model, loader, device):
    model.eval()
    batch = next(iter(loader))
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        _forward(model, batch, device)
    return _summary_embedding(model).detach().cpu()


def _model_state(model):
    module = (
        model.module if isinstance(model, DistributedDataParallel) else model)
    return module.state_dict()


def _gradient_l2_norm(parameters):
    norms = [
        parameter.grad.detach().float().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.stack(norms).norm(2))


def _parameter_snapshots(parameter_groups):
    return {
        group['name']: [
            parameter.detach().clone() for parameter in group['params']
        ]
        for group in parameter_groups
    }


def _relative_parameter_change(parameter_groups, reference):
    diagnostics = {}
    for group in parameter_groups:
        name = group['name']
        current = group['params']
        baseline = reference[name]
        difference = torch.stack([
            (parameter.detach().float() - initial.float()).square().sum()
            for parameter, initial in zip(current, baseline)
        ]).sum().sqrt()
        weight = torch.stack([
            initial.float().square().sum() for initial in baseline
        ]).sum().sqrt()
        diagnostics[name] = float(difference / weight.clamp_min(1e-12))
    return diagnostics


@torch.no_grad()
def _parameter_value_diagnostics(model, parameter_groups):
    """Summarize trainable weights and biases without changing training.

    Statistics are aggregated by optimizer group and by true parameter name:
    only parameters whose names end in ``.bias`` enter the bias bucket, so
    one-dimensional normalization scales are not accidentally called biases.
    The scan runs once per epoch rather than every optimizer step because the
    all-patch classifier can contain tens of millions of parameters.
    """
    module = (
        model.module if isinstance(model, DistributedDataParallel) else model
    )
    name_by_parameter = {
        id(parameter): name for name, parameter in module.named_parameters()
    }
    diagnostics = {}
    for group in parameter_groups:
        group_name = group['name']
        buckets = {'weight': [], 'bias': []}
        for parameter in group['params']:
            name = name_by_parameter.get(id(parameter), '')
            kind = 'bias' if name.endswith('.bias') else 'weight'
            buckets[kind].append(parameter.detach().float())

        for kind, tensors in buckets.items():
            if not tensors:
                continue
            count = sum(tensor.numel() for tensor in tensors)
            total = torch.stack([tensor.sum() for tensor in tensors]).sum()
            square_total = torch.stack([
                tensor.square().sum() for tensor in tensors
            ]).sum()
            absolute_total = torch.stack([
                tensor.abs().sum() for tensor in tensors
            ]).sum()
            absolute_max = torch.stack([
                tensor.abs().max() for tensor in tensors
            ]).max()
            nonfinite_count = torch.stack([
                (~torch.isfinite(tensor)).sum() for tensor in tensors
            ]).sum()
            zero_count = torch.stack([
                tensor.eq(0).sum() for tensor in tensors
            ]).sum()

            mean = total / count
            variance = (square_total / count - mean.square()).clamp_min(0)
            prefix = f'{group_name}/{kind}'
            diagnostics.update({
                f'{prefix}/count': int(count),
                f'{prefix}/mean': float(mean),
                f'{prefix}/std': float(variance.sqrt()),
                f'{prefix}/rms': float((square_total / count).sqrt()),
                f'{prefix}/mean_abs': float(absolute_total / count),
                f'{prefix}/max_abs': float(absolute_max),
                f'{prefix}/zero_fraction': float(zero_count / count),
                f'{prefix}/nonfinite_fraction': float(
                    nonfinite_count / count
                ),
            })
    return diagnostics


def _optimizer_updates_per_epoch(num_microbatches, accumulation_steps):
    assert num_microbatches > 0
    assert accumulation_steps > 0
    return math.ceil(num_microbatches / accumulation_steps)


def _initial_selection_score(metric):
    return float('inf') if metric == 'rmse' else float('-inf')


def _selection_improved(metric, value, best_value, has_checkpoint):
    if not has_checkpoint:
        return True
    return value < best_value if metric == 'rmse' else value > best_value


def _training_signature(config):
    signature = copy.deepcopy(config)
    signature['tracking']['resume'] = False
    signature['tracking']['run_id'] = None
    signature['tracking']['resume_checkpoint'] = None
    return canonical_sha256(signature)


def _resume_signature(config):
    """Hash training semantics while allowing execution-resource changes."""
    signature = copy.deepcopy(config)
    signature.pop('resolved', None)
    signature['tracking']['resume'] = False
    signature['tracking']['run_id'] = None
    signature['tracking']['resume_checkpoint'] = None
    signature['data'].pop('num_workers', None)
    for key in (
        'slurm_account',
        'slurm_partition',
        'slurm_exclude_nodes',
        'slurm_cpus_per_task',
        'slurm_memory',
        'slurm_time',
        'slurm_exclusive',
    ):
        signature['runtime'].pop(key, None)
    return canonical_sha256(signature)


def _parameter_groups(model, optimization):
    """Build disjoint full-fine-tuning groups with explicit learning rates."""
    if isinstance(model, DistributedDataParallel):
        model = model.module
    tokenizer = list(model.encoder.patch_encoder.parameters())
    context = list(model.encoder.positional_encoder.parameters())
    context.extend(model.encoder.context_encoder.parameters())
    head_parameters = list(model.head.parameters())
    # Attention pooling is part of the trainable downstream head but is kept
    # outside the legacy Sequential/ModuleDict classifier containers.
    classifier = getattr(model, 'classifier', None)
    groups = (
        ('tokenizer', tokenizer, optimization['tokenizer_learning_rate']),
        ('encoder', context, optimization['encoder_learning_rate']),
        ('head', head_parameters, optimization['head_learning_rate']),
    )

    parameter_groups = []
    identifiers = []
    for name, parameters, learning_rate in groups:
        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        if not trainable:
            continue
        parameter_groups.append({
            'name': name,
            'params': trainable,
            'lr': learning_rate,
        })
        identifiers.extend(id(parameter) for parameter in trainable)
    assert len(identifiers) == len(set(identifiers))
    expected = {
        id(parameter) for parameter in model.parameters()
        if parameter.requires_grad
    }
    assert set(identifiers) == expected
    return parameter_groups


def run_downstream(config):
    assert torch.cuda.is_available()
    assert 'RANK' in os.environ
    runtime = config['runtime']
    optimization = config['optimization']
    dataset_name = config['data']['dataset']
    spec = get_dataset_spec(dataset_name)
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == runtime['num_gpus']
    assert torch.cuda.device_count() == runtime['num_gpus']
    _seed_everything(config['experiment']['seed'] + rank)

    output_dir = (
        Path(runtime['output_dir'])
        / config['experiment']['trial_name']
        / dataset_name
        / config['model']['type']
        / config['model'].get('pooling', 'default')
        / f"seed-{config['experiment']['seed']}"
    )
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / 'resolved_config.yaml').open(
            'w', encoding='utf-8'
        ) as output:
            yaml.safe_dump(config, output, sort_keys=False)
    dist.barrier()

    run = None
    run_id = config['tracking']['run_id']
    if rank == 0:
        run = init_wandb(
            stage=dataset_name,
            dataset=dataset_name,
            trial_name=config['experiment']['trial_name'],
            job_type=config['tracking']['job_type'],
            config=config,
            output_dir=str(output_dir),
            entity=config['tracking']['entity'],
            run_id=run_id,
            resume=config['tracking']['resume'],
        )
        run_id = run.id

    checkpoint = torch.load(
        config['model']['checkpoint'], map_location='cpu'
    )
    tokenizer_config = checkpoint['resolved_config']['latent_tokenizer']
    assert tokenizer_config['mode'] == 'none'
    datasets = _make_datasets(
        config['resolved']['dataset_dir'],
        dataset_name,
        sample_scale_correction=config['data'].get(
            'sample_scale_correction'),
    )
    subject_cv = config['data'].get('subject_cv')
    if subject_cv is not None:
        from src.datasets.subject_cv import apply_subject_cv
        # The primary BAcc/R² checkpoint remains the only checkpoint evaluated
        # on test during training.  AUROC/Kappa selectors are saved from
        # validation for a later, explicitly requested test evaluation.
        evaluate_this_selector_on_test = (
            runtime['evaluate_test'] and (
                metric == primary_selection_metric
                or runtime.get('evaluate_comparison_test', False)
            )
        )
        if evaluate_this_selector_on_test:
            raise ValueError('LOSO folds must not evaluate test')
        datasets = apply_subject_cv(datasets, dataset_name, subject_cv)
    reproducibility = build_reproducibility_manifest(
        config,
        dataset_fingerprint={
            'split_sizes': {
                split: len(dataset)
                for split, dataset in datasets.items()
            },
            'active_channel_count': int(
                datasets['train'].channel_validity.sum()),
            'active_channel_names': [
                name for name, valid in zip(
                    datasets['train'].channel_names,
                    datasets['train'].channel_validity.tolist(),
                ) if valid
            ],
            'subject_counts': {
                split: len(set(_diagnostic_subject_ids(dataset)))
                for split, dataset in datasets.items()
                if hasattr(dataset, 'subject_ids')
            },
            'subject_overlaps': {
                'train_val': sorted(
                    set(_diagnostic_subject_ids(datasets['train']))
                    & set(_diagnostic_subject_ids(datasets['val']))
                ),
                'train_test': sorted(
                    set(_diagnostic_subject_ids(datasets['train']))
                    & set(_diagnostic_subject_ids(datasets['test']))
                ),
                'val_test': sorted(
                    set(_diagnostic_subject_ids(datasets['val']))
                    & set(_diagnostic_subject_ids(datasets['test']))
                ),
            },
        },
    )
    if rank == 0:
        write_reproducibility_manifest(
            reproducibility, output_dir / 'reproducibility.json')
    loaders = _make_loaders(
        datasets,
        optimization['batch_size_per_gpu'],
        config['data']['num_workers'],
        rank,
        world_size,
        seed=config['experiment']['seed'],
    )
    construction_rng = torch.get_rng_state()
    model = build_downstream_model(config).to(device)
    if rank == 0:
        initialization_audit = write_initialization_audit(
            {'model': model}, output_dir / 'initialization.json')
        reference_config = runtime.get('initialization_reference_config')
        if reference_config:
            from src.config import load_downstream_config
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(construction_rng)
                reference_model = build_downstream_model(load_downstream_config(reference_config))
                reference = initialization_fingerprints({'model': reference_model})
                del reference_model
            initialization_audit['comparison'] = compare_initialization(initialization_audit, reference)
            initialization_audit['reference_config'] = reference_config
            (output_dir / 'initialization.json').write_text(json.dumps(initialization_audit, indent=2) + '\n')
            if runtime.get('initialization_require_common_match', False) and not initialization_audit['comparison']['all_common_tensors_equal']:
                raise ValueError('Downstream common initialization mismatch')
    assert all(parameter.requires_grad for parameter in model.head.parameters())
    encoder_trainable = config['model'].get('transfer_mode', 'full') != 'frozen'
    assert all(parameter.requires_grad == encoder_trainable
               for parameter in model.encoder.parameters())
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )
    if rank == 0:
        updates_per_epoch = _optimizer_updates_per_epoch(
            len(loaders['train']),
            optimization['gradient_accumulation_steps'],
        )
        run.config.update({
            'dataset_split_sizes': {
                split: len(dataset) for split, dataset in datasets.items()
            },
            'parameter_count': sum(
                parameter.numel() for parameter in model.parameters()),
            'trainable_parameter_count': sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad),
            'microbatches_per_epoch': len(loaders['train']),
            'gradient_accumulation_steps': optimization[
                'gradient_accumulation_steps'
            ],
            'updates_per_epoch': updates_per_epoch,
            'total_optimization_steps': (
                optimization['epochs'] * updates_per_epoch
            ),
        }, allow_val_change=True)

    trainable = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    parameter_groups = _parameter_groups(model, optimization)
    optimizer = torch.optim.AdamW(
        parameter_groups,
        betas=tuple(optimization['adam_betas']),
        eps=optimization['adam_epsilon'],
        weight_decay=optimization['weight_decay'],
    )
    reference_parameters = _parameter_snapshots(optimizer.param_groups)
    accumulation_steps = optimization['gradient_accumulation_steps']
    updates_per_epoch = _optimizer_updates_per_epoch(
        len(loaders['train']), accumulation_steps)
    total_steps = optimization['epochs'] * updates_per_epoch
    scheduler = GroupCosineScheduler(
        optimizer,
        total_steps=total_steps,
        min_learning_rate=optimization['min_learning_rate'],
        warmup_steps=int(round(optimization.get('warmup_epochs', 0) * updates_per_epoch)),
    )
    primary_selection_metric = optimization['selection_metric']
    selection_metrics = [primary_selection_metric]
    comparison_selection_metric = optimization.get(
        'comparison_selection_metric')
    if comparison_selection_metric is not None:
        selection_metrics.append(comparison_selection_metric)
    comparison_selection_metrics = optimization.get(
        'comparison_selection_metrics', [])
    selection_metrics.extend(comparison_selection_metrics)
    if len(selection_metrics) != len(set(selection_metrics)):
        raise ValueError('selection metrics must be unique')
    best_scores = {
        metric: _initial_selection_score(metric)
        for metric in selection_metrics
    }
    best_states = {
        metric: None for metric in selection_metrics
    }
    best_epochs = {
        metric: 0 for metric in selection_metrics
    }
    best_steps = {
        metric: 0 for metric in selection_metrics
    }
    history = []
    start_epoch = 0
    global_step = 0
    early_stopping_patience = optimization.get(
        'early_stopping_patience')
    early_stopping_bad_epochs = 0
    config_sha256 = _training_signature(config)
    diagnostic_every_steps = int(runtime.get(
        'diagnostic_every_steps', 100
    ))
    if diagnostic_every_steps <= 0:
        raise ValueError('diagnostic_every_steps must be positive')
    optimizer_updates = 0
    clipped_updates = 0
    if config['tracking']['resume']:
        checkpoint = torch.load(
            config['tracking']['resume_checkpoint'], map_location='cpu')
        if checkpoint['config_sha256'] != config_sha256:
            assert (
                _resume_signature(checkpoint['resolved_config'])
                == _resume_signature(config)
            )
        assert checkpoint['wandb_run_id'] == run_id
        model.module.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_scores = checkpoint['best_scores']
        best_states = checkpoint['best_models']
        best_epochs = checkpoint['best_epochs']
        best_steps = checkpoint['best_steps']
        history = checkpoint['history']
        early_stopping_bad_epochs = checkpoint.get(
            'early_stopping_bad_epochs', 0)
        restore_rng_state(checkpoint['rng_states'][rank])

    # The reference must reflect the actual starting checkpoint.  In resume
    # mode that checkpoint is loaded above, so capture the fixed probe only
    # after restoring all training state.
    initial_probe_embeddings = _fixed_probe_embeddings(
        model, loaders['val'], device)

    for epoch in range(start_epoch, optimization.get('stop_after_epoch', optimization['epochs'])):
        loaders['train'].sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        total_examples = 0
        training_forward_diagnostics = []
        gradient_diagnostics = {}
        train_logits = []
        train_labels = []
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loaders['train']):
            group_start = (
                batch_index // accumulation_steps
            ) * accumulation_steps
            group_size = min(
                accumulation_steps,
                len(loaders['train']) - group_start,
            )
            should_update = (
                (batch_index + 1) % accumulation_steps == 0
                or batch_index + 1 == len(loaders['train'])
            )
            diagnostic_now = should_update and (
                global_step == 0
                or (global_step + 1) % diagnostic_every_steps == 0
            )
            labels = batch['label'].to(device, non_blocking=True)
            set_model_diagnostics(model, diagnostic_now)
            classifier = _classifier_module(model)
            if classifier is not None:
                classifier.diagnostics_enabled = diagnostic_now
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = _forward(model, batch, device)
                loss = _loss(
                    spec.task,
                    logits,
                    labels,
                    optimization,
                )
            metric_logits = logits.detach()
            metric_labels = labels.detach()
            if metric_labels.ndim > 1:
                metric_logits = metric_logits.reshape(
                    -1, metric_logits.shape[-1])
                metric_labels = metric_labels.reshape(-1)
            train_logits.append(metric_logits)
            train_labels.append(metric_labels)
            if diagnostic_now:
                training_forward_diagnostics.append(
                    _last_forward_diagnostics(model))
            if classifier is not None:
                classifier.diagnostics_enabled = False
            assert torch.isfinite(loss)
            step_loss_diagnostics = (
                classification_loss_diagnostics(
                    spec.task,
                    logits.detach(),
                    labels.detach(),
                    optimization['label_smoothing'],
                )
                if diagnostic_now
                else {}
            )
            (loss / group_size).backward()
            if diagnostic_now:
                detailed_gate_gradients = {
                    key: float(value) for key, value in gate_diagnostics(model).items()
                }
                pre_group_norms = {
                    group['name']: _gradient_l2_norm(group['params'])
                    for group in optimizer.param_groups
                }
                pre_update_parameters = _parameter_snapshots(
                    optimizer.param_groups
                )
            if should_update:
                pre_clip_total = torch.nn.utils.clip_grad_norm_(
                    trainable,
                    optimization['gradient_clip_norm'],
                    error_if_nonfinite=True,
                )
                optimizer_updates += 1
                if float(pre_clip_total) > optimization['gradient_clip_norm']:
                    clipped_updates += 1
                if diagnostic_now:
                    pre_clip_total = float(pre_clip_total)
                    clip_scale = min(
                        1.0,
                        optimization['gradient_clip_norm']
                        / (pre_clip_total + 1e-6),
                    )
                    gradient_diagnostics = {
                        **detailed_gate_gradients,
                        'pre_clip_total': pre_clip_total,
                        'post_clip_total': pre_clip_total * clip_scale,
                        'clip_scale': clip_scale,
                        **{
                            f'pre_clip/{name}': value
                            for name, value in pre_group_norms.items()
                        },
                        **{
                            f'post_clip/{name}': value * clip_scale
                            for name, value in pre_group_norms.items()
                        },
                    }
                learning_rates = scheduler.step()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if diagnostic_now:
                    update_ratios = _relative_parameter_change(
                        optimizer.param_groups, pre_update_parameters
                    )
                    drift_ratios = _relative_parameter_change(
                        optimizer.param_groups, reference_parameters
                    )
                    del pre_update_parameters
                    gradient_diagnostics.update({
                        **{
                            f'update_to_weight/{name}': value
                            for name, value in update_ratios.items()
                        },
                        **{
                            f'weight_drift/{name}': value
                            for name, value in drift_ratios.items()
                        },
                        'clip_applied': float(
                            pre_clip_total
                            > optimization['gradient_clip_norm']
                        ),
                        'clip_ratio_cumulative': (
                            clipped_updates / optimizer_updates),
                    })
                    if rank == 0:
                        step_forward = training_forward_diagnostics[-1]
                        diagnostic_record = {
                            'epoch': epoch + 1, 'global_step': global_step,
                            'forward': step_forward, 'gradient': gradient_diagnostics,
                            'class_conditional_unweighted_loss': {k: float(v) for k, v in step_loss_diagnostics.items()},
                            'last_microbatch_label_count': int(metric_labels.numel()),
                            'last_microbatch_class_counts': (
                                torch.bincount(metric_labels.long().cpu(), minlength=spec.num_outputs if spec.task == 'multiclass' else 2).tolist()
                                if spec.task != 'regression' else None),
                            'attention_sampling': 'up to 32 evenly spaced axial rows per encoder attention; pre-dropout, valid query-key pairs',
                        }
                        with (output_dir / 'diagnostics-steps.jsonl').open('a') as diagnostic_file:
                            diagnostic_file.write(json.dumps(diagnostic_record, sort_keys=True) + '\n')
                        run.log({
                            'train/step_loss': float(loss.detach()),
                            **{
                                f'train/class_conditional/{key}': float(value)
                                for key, value in step_loss_diagnostics.items()
                            },
                            **{
                                f'{dataset_name}/diagnostics/train_step/{key}': value
                                for key, value in step_forward.items()
                            },
                            **{
                                f'{dataset_name}/diagnostics/optimizer/{key}': value
                                for key, value in gradient_diagnostics.items()
                            },
                            **{
                                f'learning_rate/{group["name"]}': learning_rate
                                for group, learning_rate in zip(
                                    optimizer.param_groups,
                                    learning_rates,
                                )
                            },
                            'epoch': epoch + 1,
                            'global_step': global_step,
                        }, step=global_step)
            total_loss += float(loss.detach()) * labels.shape[0]
            total_examples += labels.shape[0]
        totals = torch.tensor(
            [total_loss, total_examples],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(totals)
        total_loss, total_examples = totals.tolist()

        train_metrics = downstream_metrics(
            spec.task,
            torch.cat(train_logits),
            torch.cat(train_labels),
        )

        (
            validation,
            validation_diagnostics,
            validation_subject_diagnostics,
            validation_probe_embeddings,
        ) = evaluate(
            model,
            loaders['val'],
            spec.task,
            device,
        )
        validation_diagnostics['fixed_probe_linear_cka'] = float(linear_cka(
            initial_probe_embeddings,
            validation_probe_embeddings,
        ))
        parameter_diagnostics = _parameter_value_diagnostics(
            model, optimizer.param_groups
        )
        improved_metrics = [
            metric for metric in selection_metrics
            if _selection_improved(
                metric,
                validation[metric],
                best_scores[metric],
                best_states[metric] is not None,
            )
        ]
        # A CV refit selects a precommitted epoch, never the original val/test.
        # Keep the original cosine horizon even when stopping at that epoch.
        fixed_epoch = optimization.get('cv_selected_epoch')
        if fixed_epoch is not None:
            improved_metrics = selection_metrics if epoch + 1 == fixed_epoch else []
        improved_state = None
        if improved_metrics:
            improved_state = {
                key: value.detach().cpu().clone()
                for key, value in _model_state(model).items()
            }
        for metric in selection_metrics:
            if metric in improved_metrics:
                best_scores[metric] = validation[metric]
                best_epochs[metric] = epoch + 1
                best_steps[metric] = global_step
                best_states[metric] = improved_state
        # The run preserves one checkpoint per requested selection metric.
        # Patience therefore advances only when none of those validation
        # metrics improves; monitoring only the primary metric would truncate
        # the comparison checkpoints unfairly.
        if improved_metrics:
            early_stopping_bad_epochs = 0
        else:
            early_stopping_bad_epochs += 1
        epoch_result = {
            'epoch': epoch + 1,
            'train_loss': total_loss / total_examples,
            'train': train_metrics,
            'validation': validation,
            'generalization_gap': {
                metric: float(train_metrics[metric] - validation[metric])
                for metric in validation
                if metric in train_metrics
            },
            'diagnostics': {
                'train': _mean_diagnostics(
                    training_forward_diagnostics),
                'validation': validation_diagnostics,
                'gradient': gradient_diagnostics,
                'parameters': parameter_diagnostics,
            },
            'early_stopping_bad_epochs': early_stopping_bad_epochs,
        }
        history.append(epoch_result)
        if rank == 0:
            subject_path = output_dir / (
                f'validation-subject-metrics-epoch-{epoch + 1:03d}.json'
            )
            with subject_path.open('w', encoding='utf-8') as output:
                json.dump(
                    validation_subject_diagnostics,
                    output,
                    indent=2,
                    sort_keys=True,
                )
                output.write('\n')
            run.log({
                'train/loss': epoch_result['train_loss'],
                **{
                    f'{dataset_name}/train/{key}': value
                    for key, value in train_metrics.items()
                },
                **{
                    f'{dataset_name}/validation/{key}': value
                    for key, value in validation.items()
                },
                **{
                    f'{dataset_name}/generalization_gap/{key}': value
                    for key, value in epoch_result[
                        'generalization_gap'].items()
                },
                **{
                    f'learning_rate/{group["name"]}': learning_rate
                    for group, learning_rate in zip(
                        optimizer.param_groups, learning_rates)
                },
                **{
                    f'{dataset_name}/diagnostics/train/{key}': value
                    for key, value in epoch_result[
                        'diagnostics']['train'].items()
                },
                **{
                    f'{dataset_name}/diagnostics/validation/{key}': value
                    for key, value in validation_diagnostics.items()
                },
                **{
                    f'{dataset_name}/diagnostics/gradient/{key}': value
                    for key, value in gradient_diagnostics.items()
                },
                **{
                    f'{dataset_name}/diagnostics/parameters/{key}': value
                    for key, value in parameter_diagnostics.items()
                },
                'epoch': epoch + 1,
                'global_step': global_step,
            }, step=global_step)
            print(json.dumps({
                **epoch_result,
                'global_step': global_step,
                'best_epochs': best_epochs,
                'best_scores': best_scores,
            }, sort_keys=True), flush=True)

        rng_state = capture_rng_state()
        rng_states = [None] * world_size
        dist.all_gather_object(rng_states, rng_state)
        if rank == 0:
            atomic_save({
                'model': _model_state(model),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler_state': None,
                'epoch': epoch + 1,
                'global_step': global_step,
                'best_scores': best_scores,
                'best_models': best_states,
                'best_epochs': best_epochs,
                'best_steps': best_steps,
                'history': history,
                'early_stopping_bad_epochs': early_stopping_bad_epochs,
                'rng_states': rng_states,
                'wandb_run_id': run_id,
                'config_sha256': config_sha256,
                'resolved_config': config,
            }, output_dir / 'last.pth')
        dist.barrier()
        if (
            early_stopping_patience is not None
            and early_stopping_bad_epochs >= early_stopping_patience
        ):
            if rank == 0:
                print(json.dumps({
                    'early_stopping': True,
                    'patience': early_stopping_patience,
                    'best_epochs': best_epochs,
                    'stopped_after_epoch': epoch + 1,
                }, sort_keys=True), flush=True)
            break

    module = model.module
    selection_results = {}
    for metric in selection_metrics:
        module.load_state_dict(best_states[metric], strict=True)
        best_validation_subject_path = output_dir / (
            'validation-subject-metrics-epoch-'
            f'{best_epochs[metric]:03d}.json'
        )
        with best_validation_subject_path.open(
            encoding='utf-8'
        ) as source:
            best_validation_subject_diagnostics = json.load(source)
        test_metrics = None
        if runtime['evaluate_test']:
            (
                test_metrics,
                test_diagnostics,
                test_subject_diagnostics,
                _,
            ) = evaluate(
                model,
                loaders['test'],
                spec.task,
                device,
            )
        else:
            test_diagnostics = None
            test_subject_diagnostics = None
        selection_results[metric] = {
            'best_epoch': best_epochs[metric],
            'best_step': best_steps[metric],
            'best_validation': history[
                best_epochs[metric] - 1]['validation'],
            'best_validation_subject_diagnostics': (
                best_validation_subject_diagnostics),
            'test': test_metrics,
            'test_diagnostics': test_diagnostics,
            'test_subject_diagnostics': test_subject_diagnostics,
        }
    primary_result = selection_results[primary_selection_metric]
    result = {
        'trial_name': config['experiment']['trial_name'],
        'dataset': dataset_name,
        'model': config['model']['type'],
        'transfer_mode': config['model'].get('transfer_mode', 'full'),
        'encoder_initialization': (
            'random_architecture_from_checkpoint' if
            config['model'].get('transfer_mode') == 'random' else 'pretrained'),
        'pooling': config['model'].get('pooling'),
        'seed': config['experiment']['seed'],
        'best_epoch': primary_result['best_epoch'],
        'best_step': primary_result['best_step'],
        'epochs_ran': len(history),
        'selection_metric': primary_selection_metric,
        'comparison_selection_metric': comparison_selection_metric,
        'comparison_selection_metrics': comparison_selection_metrics,
        'best_validation': primary_result['best_validation'],
        'best_validation_diagnostics': history[
            primary_result['best_epoch'] - 1]['diagnostics']['validation'],
        'test': primary_result['test'],
        'test_diagnostics': primary_result['test_diagnostics'],
        'selection_results': selection_results,
        'lineage': config['lineage'],
        'source_tree_sha256': reproducibility['source_tree_sha256'],
    }
    if subject_cv is not None or optimization.get('cv_selected_epoch') is not None:
        result['validation_protocol'] = {
            'fold': subject_cv,
            'cv_selected_epoch': optimization.get('cv_selected_epoch'),
            'selection_report': optimization.get('cv_selection_report'),
            'scheduler_horizon_epochs': optimization['epochs'],
        }
    if rank == 0:
        if subject_cv is not None:
            with (output_dir / 'validation_history.json').open('w', encoding='utf-8') as output:
                json.dump([{'epoch': row['epoch'], 'validation': row['validation']}
                           for row in history], output, indent=2)
        checkpoint_paths = {}
        for metric in selection_metrics:
            checkpoint_path = output_dir / f'best-{metric}.pth'
            atomic_save(
                {
                    'model': best_states[metric],
                    'selection_metric': metric,
                    'selection_result': selection_results[metric],
                    'config': config,
                    'reproducibility': reproducibility,
                },
                checkpoint_path,
            )
            checkpoint_paths[metric] = checkpoint_path
        result_path = output_dir / 'result.json'
        with result_path.open('w', encoding='utf-8') as output:
            json.dump(result, output, indent=2, sort_keys=True)
            output.write('\n')
        import wandb
        artifact = wandb.Artifact(
            (
                f"{config['experiment']['trial_name']}__"
                f"{dataset_name}__{config['model']['type']}__"
                f"{config['model'].get('pooling', 'default')}__best"
            ),
            type='model',
            metadata=result,
        )
        for checkpoint_path in checkpoint_paths.values():
            artifact.add_file(str(checkpoint_path))
        artifact.add_file(str(output_dir / 'resolved_config.yaml'))
        artifact.add_file(str(output_dir / 'reproducibility.json'))
        artifact.add_file(str(result_path))
        run.log_artifact(artifact)
        for metric, metric_result in selection_results.items():
            best_validation_diagnostics = history[
                metric_result['best_epoch'] - 1
            ]['diagnostics']['validation']
            run.log({
                **{
                    f'{dataset_name}/{metric}/best_validation/{key}': value
                    for key, value
                    in metric_result['best_validation'].items()
                },
                **{
                    f'{dataset_name}/{metric}/test/{key}': value
                    for key, value in (metric_result['test'] or {}).items()
                },
                **{
                    f'{dataset_name}/{metric}/best_validation_diagnostics/{key}': value
                    for key, value in best_validation_diagnostics.items()
                },
                **{
                    f'{dataset_name}/{metric}/test_diagnostics/{key}': value
                    for key, value in (
                        metric_result['test_diagnostics'] or {}).items()
                },
            })
        run.summary.update({
            'best_epochs': best_epochs,
            'best_steps': best_steps,
            'best_validation_scores': best_scores,
            'last_epoch': len(history),
            'last_step': global_step,
            'checkpoint_best_paths': {
                metric: str(path)
                for metric, path in checkpoint_paths.items()
            },
        })
        run.finish()
        (output_dir / 'last.pth').unlink(missing_ok=True)
    dist.barrier()
    dist.destroy_process_group()
    return result


if __name__ == "__main__":
    from src.config import load_downstream_config

    run_downstream(load_downstream_config(_parse_args().config))
