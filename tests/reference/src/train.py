"""Distributed raw-patch masked autoencoder pretraining for EEG."""

import argparse
from contextlib import nullcontext
import copy
import json
import logging
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import yaml

from src.datasets.pretraining_dataset import make_pretraining_loader
from src.models.factory import build_models
from src.utils.checkpointing import (
    atomic_save,
    capture_rng_state,
    load_pretrain_checkpoint,
    unwrap,
)
from src.utils.optimization import build_pretrain_optimizer
from src.utils.model_diagnostics import (
    set_model_diagnostics, collect_model_diagnostics, gate_diagnostics,
    write_initialization_audit,
    initialization_fingerprints,
)
from src.utils.masking import build_masking_policy
from src.utils.reconstruction import (
    gather_target_blocks,
    raw_patch_grid,
    reconstruction_diagnostics,
    representation_diagnostics,
)
from src.utils.reconstruction_loss import reconstruction_losses
from src.tracking import init_wandb
from src.utils.reproducibility import (
    build_reproducibility_manifest,
    canonical_sha256,
    write_reproducibility_manifest,
)


LOGGER = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _training_signature(config):
    signature = copy.deepcopy(config)
    signature['tracking']['resume'] = False
    signature['tracking']['run_id'] = None
    signature['tracking']['resume_checkpoint'] = None
    return canonical_sha256(signature)


def _save_checkpoint(
    path,
    context_encoder,
    decoder,
    optimizer,
    lr_scheduler,
    epoch,
    global_step,
    config,
    config_sha256,
    dataset_fingerprint,
    reproducibility,
    rng_states,
    wandb_run_id,
    best_epoch,
    best_step,
    best_train_epoch_loss,
):
    atomic_save({
        'context_encoder': unwrap(context_encoder).state_dict(),
        'decoder': unwrap(decoder).state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'scaler_state': None,
        'epoch': epoch,
        'global_step': global_step,
        'resolved_config': config,
        'config_sha256': config_sha256,
        'dataset_fingerprint': dataset_fingerprint,
        'reproducibility': reproducibility,
        'rng_states': rng_states,
        'wandb_run_id': wandb_run_id,
        'best_epoch': best_epoch,
        'best_step': best_step,
        'best_train_epoch_loss': best_train_epoch_loss,
    }, path)


def _reduce_mean(value, device, world_size):
    tensor = torch.tensor(float(value), device=device)
    dist.all_reduce(tensor)
    return float(tensor / world_size)


def _gradient_norm(parameters):
    squared = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squared:
        return 0.0
    return float(torch.stack(squared).sum().sqrt())


def main(config):
    assert torch.cuda.is_available()
    assert 'RANK' in os.environ
    runtime = config['runtime']
    optimization = config['optimization']
    data = config['data']
    reconstruction = config['mae']
    branch_auxiliary = reconstruction.get('branch_auxiliary', {})
    branch_auxiliary_enabled = bool(branch_auxiliary.get('enabled', False))
    auxiliary_lambda = float(branch_auxiliary.get('lambda_star', 0.0))
    assert reconstruction['objective'] == 'raw_patch_mae'

    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == runtime['num_gpus']
    assert torch.cuda.device_count() == 1

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.ERROR,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    seed = config['experiment']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(
        runtime['deterministic'], warn_only=False)
    torch.backends.cudnn.deterministic = runtime['deterministic']
    torch.backends.cudnn.benchmark = not runtime['deterministic']

    trial_name = config['experiment']['trial_name']
    output_dir = Path(runtime['output_dir']) / trial_name
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
            stage='pretrain',
            dataset=data['dataset'],
            trial_name=trial_name,
            job_type=config['tracking']['job_type'],
            config=config,
            output_dir=str(output_dir),
            entity=config['tracking']['entity'],
            run_id=run_id,
            resume=config['tracking']['resume'],
        )
        run_id = run.id

    channels = len(data['channel_names'])
    num_patches = config['resolved']['num_patches']
    patch_samples = config['patch_encoder']['patch_samples']
    dataset, loader, sampler = make_pretraining_loader(
        dataset_dir=config['resolved']['dataset_dir'],
        batch_size=optimization['batch_size_per_gpu'],
        pin_mem=data['pin_memory'],
        num_workers=data['num_workers'],
        world_size=world_size,
        rank=rank,
        seed=seed,
        drop_last=True,
        channels=channels,
        num_patches=num_patches,
        patch_samples=patch_samples,
    )
    first_signal, _ = dataset[0]
    assert first_signal.shape == (
        channels, config['resolved']['segment_samples'])
    assert torch.isfinite(first_signal).all()
    dataset.close()

    accumulation_steps = optimization['gradient_accumulation_steps']
    training_microbatches = (
        len(loader) // accumulation_steps) * accumulation_steps
    assert training_microbatches > 0
    updates_per_epoch = training_microbatches // accumulation_steps
    total_updates = optimization['epochs'] * updates_per_epoch

    factory_cpu_rng = torch.get_rng_state()
    factory_cuda_rng = torch.cuda.get_rng_state(device)
    context_encoder, decoder = build_models(config, device)
    initialization_reference = None
    if runtime.get('initialization_reference_config'):
        from src.config import load_pretrain_config
        with torch.random.fork_rng(devices=[device.index]):
            torch.set_rng_state(factory_cpu_rng)
            torch.cuda.set_rng_state(factory_cuda_rng, device)
            reference_encoder, reference_decoder = build_models(
                load_pretrain_config(runtime['initialization_reference_config']), device)
            initialization_reference = initialization_fingerprints(
                {'encoder': reference_encoder, 'decoder': reference_decoder})
            del reference_encoder, reference_decoder
    initialization_audit = write_initialization_audit(
        {'encoder': context_encoder, 'decoder': decoder},
        output_dir / f'initialization-rank{rank}.json',
        reference_path=runtime.get('initialization_reference_manifest'),
        require_match=runtime.get('initialization_require_common_match', False),
        reference_report=initialization_reference,
    )
    optimizer, lr_scheduler = build_pretrain_optimizer(
        context_encoder, decoder, updates_per_epoch, config)
    context_encoder = DistributedDataParallel(
        context_encoder,
        device_ids=[local_rank],
        output_device=local_rank,
    )
    decoder = DistributedDataParallel(
        decoder,
        device_ids=[local_rank],
        output_device=local_rank,
    )
    if rank == 0:
        parameters = list(context_encoder.parameters()) + list(
            decoder.parameters())
        run.config.update({
            'dataset_sample_count': len(dataset),
            'parameter_count': sum(
                parameter.numel() for parameter in parameters),
            'trainable_parameter_count': sum(
                parameter.numel() for parameter in parameters
                if parameter.requires_grad),
            'updates_per_epoch': updates_per_epoch,
            'total_optimization_steps': total_updates,
            'initialization_comparison': initialization_audit['comparison'],
        }, allow_val_change=True)
    # Model initialization is identical across ranks; stochastic masks and
    # dropout are rank-specific during training.
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

    masking_policy = build_masking_policy(config['masking'])
    masking_generator = torch.Generator(device='cpu')
    config_sha256 = _training_signature(config)
    reproducibility = build_reproducibility_manifest(
        config, dataset_fingerprint=dataset.fingerprint)
    if rank == 0:
        write_reproducibility_manifest(
            reproducibility, output_dir / 'reproducibility.json')

    start_epoch = 0
    global_step = 0
    best_epoch = 0
    best_step = 0
    best_train_epoch_loss = float('inf')
    if config['tracking']['resume']:
        checkpoint = load_pretrain_checkpoint(
            config['tracking']['resume_checkpoint'],
            context_encoder,
            decoder,
            optimizer,
            lr_scheduler,
            config_sha256,
            dataset.fingerprint,
            rank,
        )
        assert checkpoint['wandb_run_id'] == run_id
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_epoch = checkpoint['best_epoch']
        best_step = checkpoint['best_step']
        best_train_epoch_loss = checkpoint['best_train_epoch_loss']
    assert start_epoch < optimization['epochs']

    use_bfloat16 = optimization['mixed_precision'] == 'bf16'
    valid_token_mask = unwrap(context_encoder).valid_token_mask(
        batch_size=optimization['batch_size_per_gpu'],
        num_patches=num_patches,
    )
    masking_channel_coordinates = (
        unwrap(context_encoder).default_channel_coordinates.detach().cpu()
    )
    assert valid_token_mask.any(dim=1).all()
    metrics_path = output_dir / 'metrics.jsonl'

    training_start = time.perf_counter()
    final_epoch_loss = float('nan')
    diagnostic_updates = clipped_updates = 0
    for epoch in range(start_epoch, optimization['epochs']):
        epoch_start = time.perf_counter()
        sampler.set_epoch(epoch)
        loader.generator.manual_seed(seed + epoch)
        masking_generator.manual_seed(
            seed + epoch * world_size + rank
        )
        if hasattr(masking_policy, 'set_epoch'):
            masking_policy.set_epoch(epoch)
        context_encoder.train()
        decoder.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0

        for microbatch, (signals, sample_indices) in enumerate(loader):
            if microbatch == training_microbatches:
                break
            signals = signals.to(device, non_blocking=True)
            signals = signals / data['value_scale']
            masks = masking_policy(
                batch_size=signals.shape[0],
                num_latents=valid_token_mask.shape[1],
                num_patches=num_patches,
                valid_token_mask=valid_token_mask,
                channel_coordinates=masking_channel_coordinates,
                raw_signals=signals,
                sample_rate=data['sample_rate'],
                patch_samples=patch_samples,
                generator=masking_generator,
                sample_indices=sample_indices,
            )
            update_now = (microbatch + 1) % accumulation_steps == 0
            diagnostic_now = update_now and (
                global_step == 0 or (global_step + 1) % runtime['log_every_steps'] == 0)
            set_model_diagnostics(context_encoder, diagnostic_now)
            set_model_diagnostics(decoder, diagnostic_now)
            sync_context = (
                nullcontext() if update_now else context_encoder.no_sync())
            sync_decoder = (
                nullcontext() if update_now else decoder.no_sync())

            with sync_context, sync_decoder:
                with torch.cuda.amp.autocast(
                    enabled=use_bfloat16, dtype=torch.bfloat16
                ):
                    encoder_result = context_encoder(
                        signals,
                        visible_mask=masks['context_mask'],
                        return_auxiliary=True,
                        return_branch_outputs=branch_auxiliary_enabled,
                    )
                    if branch_auxiliary_enabled:
                        context, context_valid_mask, auxiliary, branch_outputs = encoder_result
                    else:
                        context, context_valid_mask, auxiliary = encoder_result
                    context_slot_mask = (
                        unwrap(context_encoder).expand_region_mask(
                            masks['context_mask'])
                        & context_valid_mask
                    )
                    target_slot_mask = unwrap(
                        context_encoder
                    ).expand_region_mask(masks['target_mask'])
                    with torch.no_grad():
                        target_grid = raw_patch_grid(
                            signals, patch_samples
                        )
                        target, gathered_target_valid = gather_target_blocks(
                            target_grid,
                            masks['target_blocks'],
                            return_token_valid=True,
                        )
                        assert torch.equal(
                            gathered_target_valid,
                            masks['target_token_valid'],
                        )
                    if branch_auxiliary_enabled:
                        # Batch-axis packing preserves sample isolation inside
                        # the shared decoder.  Masks and block metadata follow
                        # exactly the same main/s2t/t2s order.
                        packed_context = torch.cat(
                            (branch_outputs['main'], branch_outputs['s2t'],
                             branch_outputs['t2s']), dim=0)
                        repeat = lambda value: value.repeat((3,) + (1,) * (value.ndim - 1))
                        packed_prediction = decoder(
                            packed_context, repeat(context_slot_mask),
                            repeat(target_slot_mask),
                            target_blocks=repeat(masks['target_blocks']),
                            target_block_valid=repeat(masks['target_block_valid']),
                            target_token_valid=repeat(masks['target_token_valid']),
                        )
                        prediction, prediction_s2t, prediction_t2s = packed_prediction.chunk(3, dim=0)
                    else:
                        prediction = decoder(
                            context, context_slot_mask, target_slot_mask,
                            target_blocks=masks['target_blocks'],
                            target_block_valid=masks['target_block_valid'],
                            target_token_valid=masks['target_token_valid'],
                        )
                    loss_mask = masks['target_token_valid'].flatten(
                        1
                    ).unsqueeze(-1)
                    loss_components = reconstruction_losses(
                        prediction,
                        target,
                        masks['target_token_valid'],
                        waveform_loss=reconstruction['loss'],
                        smooth_l1_beta=reconstruction['smooth_l1_beta'],
                        frequency_loss_weight=(
                            reconstruction['frequency_loss_weight']),
                        phase_loss_weight=(
                            reconstruction['phase_loss_weight']),
                        spectral_epsilon=reconstruction['spectral_epsilon'],
                    )
                    main_loss = loss_components['waveform_loss']
                    loss = loss_components['total_loss']
                    if branch_auxiliary_enabled:
                        s2t_components = reconstruction_losses(
                            prediction_s2t, target, masks['target_token_valid'],
                            waveform_loss=reconstruction['loss'], smooth_l1_beta=reconstruction['smooth_l1_beta'])
                        t2s_components = reconstruction_losses(
                            prediction_t2s, target, masks['target_token_valid'],
                            waveform_loss=reconstruction['loss'], smooth_l1_beta=reconstruction['smooth_l1_beta'])
                        auxiliary_mean = (s2t_components['waveform_loss'] + t2s_components['waveform_loss']) / 2
                        loss = loss + auxiliary_lambda * auxiliary_mean
                    else:
                        auxiliary_mean = main_loss.new_zeros(())
                        s2t_components = t2s_components = None
                    valid_prediction = prediction[loss_mask.expand_as(
                        prediction
                    )].reshape(1, -1, prediction.shape[-1])
                    valid_target = target[loss_mask.expand_as(
                        target
                    )].reshape(1, -1, target.shape[-1])
                assert torch.isfinite(loss)
                (loss / accumulation_steps).backward()

            epoch_loss += float(loss)
            gradient_norm = 0.0
            temporal_gradient_norm = 0.0
            frequency_gradient_norm = 0.0
            learning_rate = optimizer.param_groups[0]['lr']
            learning_rate_used = learning_rate
            detailed_gradients = {}
            if update_now:
                if diagnostic_now:
                    detailed_gradients = {
                        **{f'encoder/{k}': v for k, v in gate_diagnostics(unwrap(context_encoder)).items()},
                        **{f'decoder/{k}': v for k, v in gate_diagnostics(unwrap(decoder)).items()},
                    }
                encoder = unwrap(context_encoder)
                temporal_gradient_norm = _gradient_norm(
                    encoder.patch_encoder.temporal_parameters()
                )
                frequency_gradient_norm = _gradient_norm(
                    encoder.patch_encoder.frequency_parameters()
                )
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(
                    list(context_encoder.parameters())
                    + list(decoder.parameters()),
                    optimization['gradient_clip_norm'],
                    error_if_nonfinite=True,
                ))
                diagnostic_updates += 1
                clipped_updates += int(gradient_norm > optimization['gradient_clip_norm'])
                if (
                    config['experiment'].get('profile')
                    == 'historical_d192_reproduction'
                    or optimization.get('warmup_epochs', 0) > 0
                ):
                    # The archived loop advanced its warmup schedule before
                    # applying each optimizer update.
                    lr_scheduler.step()
                    learning_rate_used = optimizer.param_groups[0]['lr']
                    optimizer.step()
                else:
                    optimizer.step()
                    lr_scheduler.step()
                learning_rate = optimizer.param_groups[0]['lr']
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if diagnostic_now:
                detailed_model = {
                    **{f'encoder/{k}': v for k, v in collect_model_diagnostics(unwrap(context_encoder)).items()},
                    **{f'decoder/{k}': v for k, v in collect_model_diagnostics(unwrap(decoder)).items()},
                    **detailed_gradients,
                }
                reduced_detailed = {
                    key: _reduce_mean(value, device, world_size)
                    for key, value in detailed_model.items()
                }
                diagnostics = representation_diagnostics(valid_target)
                context_values = context[context_slot_mask]
                context_counts = context_slot_mask.sum(dim=(1, 2))
                context_means = (
                    (context * context_slot_mask.unsqueeze(-1)).sum(dim=(1, 2))
                    / context_counts[:, None]
                )
                representation_metrics = {
                    'context_std': context_values.float().std(
                        unbiased=False),
                    'target_std': valid_target.float().std(unbiased=False),
                    'prediction_std': valid_prediction.float().std(
                        unbiased=False),
                    'prediction_target_cosine': F.cosine_similarity(
                        valid_prediction.float(), valid_target.float(), dim=-1
                    ).mean(),
                }
                reconstruction_metrics = reconstruction_diagnostics(
                    valid_prediction, valid_target, data['sample_rate'])
                reduced_loss = _reduce_mean(loss, device, world_size)
                reduced_main_loss = _reduce_mean(
                    main_loss, device, world_size)
                reduced_loss_components = {
                    key: _reduce_mean(value, device, world_size)
                    for key, value in loss_components.items()
                }
                feature_variance = _reduce_mean(
                    diagnostics['feature_variance'], device, world_size)
                mean_cosine = _reduce_mean(
                    diagnostics['mean_pairwise_cosine'], device, world_size)
                reduced_gradient_norm = _reduce_mean(
                    gradient_norm, device, world_size)
                reduced_temporal_gradient = _reduce_mean(
                    temporal_gradient_norm, device, world_size)
                reduced_frequency_gradient = _reduce_mean(
                    frequency_gradient_norm, device, world_size)
                reduced_auxiliary = {
                    key: _reduce_mean(value, device, world_size)
                    for key, value in auxiliary.items()
                }
                reduced_representation = {
                    key: _reduce_mean(value, device, world_size)
                    for key, value in representation_metrics.items()
                }
                reduced_reconstruction = {
                    key: _reduce_mean(value, device, world_size)
                    for key, value in reconstruction_metrics.items()
                }
                reduced_masking = {
                    key: _reduce_mean(value, device, world_size)
                    for key, value in masks.get(
                        'masking_diagnostics', {}
                    ).items()
                }
                if rank == 0:
                    elapsed = time.perf_counter() - epoch_start
                    metrics = {
                        'train/total_loss': reduced_loss,
                        'train/main_loss': reduced_main_loss,
                        'train/reconstruction_loss': reduced_main_loss,
                        'train/waveform_loss': (
                            reduced_loss_components['waveform_loss']),
                        'train/frequency_loss': (
                            reduced_loss_components['frequency_loss']),
                        'train/weighted_frequency_loss': (
                            reduced_loss_components[
                                'weighted_frequency_loss']),
                        'train/phase_loss': (
                            reduced_loss_components['phase_loss']),
                        'train/weighted_phase_loss': (
                            reduced_loss_components['weighted_phase_loss']),
                        **({
                            'train/waveform_mse': reduced_main_loss,
                        } if reconstruction['loss'] == 'mse' else {}),
                        'train/learning_rate': learning_rate,
                        'train/learning_rate_used': learning_rate_used,
                        'train/clip_applied': float(gradient_norm > optimization['gradient_clip_norm']),
                        'train/clip_fraction_since_resume': clipped_updates / diagnostic_updates,
                        'train/gradient_norm_post_clip': min(gradient_norm, optimization['gradient_clip_norm']),
                        **{f'diagnostics/{key}': value for key, value in reduced_detailed.items()},
                        'train/gradient_norm': reduced_gradient_norm,
                        'branch/temporal_gradient_norm': (
                            reduced_temporal_gradient),
                        'branch/frequency_gradient_norm': (
                            reduced_frequency_gradient),
                        'train/feature_variance': feature_variance,
                        'train/mean_pairwise_cosine': mean_cosine,
                        **{
                            (
                                f'branch/{key}'
                                if key.startswith((
                                    'temporal_', 'frequency_'))
                                else (
                                    f'patch/{key[len("patch_"):]}'
                                    if key.startswith('patch_')
                                    else f'latent/{key}'
                                )
                            ): value
                            for key, value in reduced_auxiliary.items()
                        },
                        'train/mask_ratio': float(
                            masks['target_mask'].sum()
                            / masks['valid_token_mask'].sum()
                        ),
                        'train/context_token_count': float(
                            masks['context_mask'].sum(dim=(1, 2)).float().mean()
                        ),
                        'train/target_token_count': float(
                            masks['target_mask'].sum(dim=(1, 2)).float().mean()
                        ),
                        'train/observation_token_count': float(
                            masks['observation_mask'].sum(
                                dim=(1, 2)).float().mean()
                        ),
                        'train/observation_ratio': float(
                            masks['observation_mask'].sum()
                            / masks['valid_token_mask'].sum()
                        ),
                        'train/context_ratio': float(
                            masks['context_mask'].sum()
                            / masks['valid_token_mask'].sum()
                        ),
                        **{
                            f'masking/{key}': value
                            for key, value in reduced_masking.items()
                        },
                        **{
                            f'representation/{key}': value
                            for key, value in reduced_representation.items()
                        },
                        **{
                            f'reconstruction/{key}': value
                            for key, value in reduced_reconstruction.items()
                        },
                        'representation/latent_mean_variance': (
                            feature_variance),
                        'representation/latent_mean_absolute_cosine': (
                            reduced_auxiliary[
                                'mean_absolute_cosine_similarity'
                            ]
                        ),
                        'train/effective_global_batch_size': (
                            config['resolved']['effective_global_batch_size']),
                        'train/throughput_samples_per_second': (
                            (microbatch + 1)
                            * optimization['batch_size_per_gpu']
                            * world_size
                            / elapsed
                        ),
                        'train/examples_per_second': (
                            (microbatch + 1)
                            * optimization['batch_size_per_gpu']
                            * world_size
                            / elapsed
                        ),
                        'train/steps_per_second': (
                            ((microbatch + 1) // accumulation_steps)
                            / elapsed
                        ),
                        'train/peak_gpu_memory_bytes': (
                            torch.cuda.max_memory_allocated(device)),
                        'train/peak_gpu_memory_gb': (
                            torch.cuda.max_memory_allocated(device) / 1024**3),
                        'epoch': epoch + 1,
                        'global_step': global_step,
                    }
                    run.log(metrics, step=global_step)
                    with metrics_path.open('a', encoding='utf-8') as output:
                        output.write(json.dumps(metrics, sort_keys=True) + '\n')
                    LOGGER.info(
                        'epoch=%d step=%d loss=%.6f variance=%.6e '
                        'cosine=%.4f grad=%.4f memory=%.1fMiB',
                        epoch + 1,
                        global_step,
                        reduced_loss,
                        feature_variance,
                        mean_cosine,
                        reduced_gradient_norm,
                        torch.cuda.max_memory_allocated(device) / 1024**2,
                    )

        rng_state = capture_rng_state()
        rng_states = [None] * world_size
        dist.all_gather_object(rng_states, rng_state)
        reduced_epoch_loss = _reduce_mean(
            epoch_loss / training_microbatches, device, world_size)
        final_epoch_loss = reduced_epoch_loss
        improved = reduced_epoch_loss < best_train_epoch_loss
        if improved:
            best_train_epoch_loss = reduced_epoch_loss
            best_epoch = epoch + 1
            best_step = global_step
        save_periodic = (
            (epoch + 1) % runtime['save_every_epochs'] == 0
            or epoch + 1 == optimization['epochs']
        )
        periodic_path = (
            output_dir / f'checkpoint-epoch-{epoch + 1:04d}.pth')
        if rank == 0:
            last_path = output_dir / 'last.pth'
            _save_checkpoint(
                last_path,
                context_encoder,
                decoder,
                optimizer,
                lr_scheduler,
                epoch + 1,
                global_step,
                config,
                config_sha256,
                dataset.fingerprint,
                reproducibility,
                rng_states,
                run_id,
                best_epoch,
                best_step,
                best_train_epoch_loss,
            )
            if save_periodic:
                _save_checkpoint(
                    periodic_path,
                    context_encoder,
                    decoder,
                    optimizer,
                    lr_scheduler,
                    epoch + 1,
                    global_step,
                    config,
                    config_sha256,
                    dataset.fingerprint,
                    reproducibility,
                    rng_states,
                    run_id,
                    best_epoch,
                    best_step,
                    best_train_epoch_loss,
                )
                checkpoints = sorted(
                    output_dir.glob('checkpoint-epoch-*.pth'))
                for old_checkpoint in checkpoints[
                    :-runtime['keep_last_checkpoints']
                ]:
                    old_checkpoint.unlink()
            run.log({
                'train/epoch_loss': reduced_epoch_loss,
                'train/epoch_duration_seconds': (
                    time.perf_counter() - epoch_start),
                'epoch': epoch + 1,
                'global_step': global_step,
            }, step=global_step)
        dist.barrier()

    if rank == 0:
        import wandb
        final_path = output_dir / 'last.pth'
        final_artifact = wandb.Artifact(
            name=(
                f"{trial_name}__{config['tracking']['job_type']}__final"
            ),
            type='model',
            metadata={
                'trial_name': trial_name,
                'job_type': config['tracking']['job_type'],
                'seed': seed,
                'epoch': optimization['epochs'],
                'step': global_step,
                'validation_loss': None,
                'git_commit': None,
                'config_sha256': config_sha256,
                'source_tree_sha256': (
                    reproducibility['source_tree_sha256']),
                'tokenizer_config': config['latent_tokenizer'],
                'model_config': config['encoder'],
            },
        )
        final_artifact.add_file(str(final_path))
        final_artifact.add_file(str(output_dir / 'resolved_config.yaml'))
        final_artifact.add_file(str(output_dir / 'reproducibility.json'))
        run.log_artifact(final_artifact)
        run.summary.update({
            'best_epoch': best_epoch,
            'best_step': best_step,
            'best_val_loss': None,
            'best_train_epoch_loss': best_train_epoch_loss,
            'last_epoch': optimization['epochs'],
            'last_step': global_step,
            'final_train_loss': final_epoch_loss,
            'checkpoint_last_path': str(output_dir / 'last.pth'),
            'checkpoint_final_path': str(final_path),
            'total_training_time': time.perf_counter() - training_start,
            'peak_gpu_memory_gb': (
                torch.cuda.max_memory_allocated(device) / 1024**3),
        })
        run.finish()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    from src.config import load_pretrain_config

    main(load_pretrain_config(_parse_args().config))
