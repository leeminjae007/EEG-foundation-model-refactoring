"""Batch → model → loss → backward → clipping → optimizer 순서의 학습 루프."""

from contextlib import nullcontext
import json
from pathlib import Path
import math
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from src.data.datasets.pretraining_dataset import make_pretraining_loader
from src.data.datasets.registry import get_dataset_spec
from src.model import EEGEncoder, FinetuneModel, PretrainModel, SleepModel
from src.modules.loss import downstream_loss, reconstruction_loss
from src.modules.masking import gather_targets, make_masks
from src.training.checkpoint import load_checkpoint, rng_state, save_checkpoint
from src.training.diagnostics import (capture, fingerprint, measurements, person_id,
                                      prediction_report, probe_report, reconstruction)
from src.training.metrics import metrics
from src.training.runtime import ROOT, setup
from src.training.scheduler import GroupCosineScheduler
from src.training.pretrain_rng import initialize_pretrain_rng, pretrain_rng_mode, pretrain_rng_report
from src.training.smoke import TimedSmoke, LimitedLoader


def autocast(device):
    # BF16에는 GradScaler를 쓰지 않는다. CPU는 소량 검증용 FP32이다.
    if device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    return nullcontext()


def append_record(path, record):
    with path.open("a") as output:
        output.write(json.dumps(record) + "\n")


def collect_rng(device, world):
    state = rng_state(device)
    if world == 1:
        return [state]
    states = [None] * world
    dist.all_gather_object(states, state)
    return states


def run_pretrain(config, args):
    device, rank, world = setup(args.device, args.distributed, config["seed"],
                               config["runtime"]["deterministic"],
                               not config["runtime"]["deterministic"], False)
    data = config["data"]
    optimization = config["optimization"]
    smoke_seconds = getattr(args, "smoke_seconds", 0)
    partial_smoke = args.smoke or bool(smoke_seconds)
    timer = TimedSmoke(smoke_seconds, device, world)
    if "warmup_epochs" in optimization or "warmup_ratio" in optimization:
        raise ValueError("Downstream warmup is not supported; remove the warmup setting")
    output = ROOT / config["runtime"]["output"]
    if args.smoke:
        output = ROOT / "outputs/smoke/pretrain"
        if config["masking"].get("policy") == "geometry_tubelet":
            output = ROOT / "outputs/smoke" / Path(config["runtime"]["output"]).name
    output.mkdir(parents=True, exist_ok=True)
    batch_size = optimization["batch_size_per_gpu"]
    workers = data["num_workers"]
    accumulation = optimization["gradient_accumulation_steps"]
    if args.smoke:
        batch_size, workers, accumulation = 2, 0, 1
    dataset, loader, sampler = make_pretraining_loader(
        dataset_dir=data["dataset_dir"], batch_size=batch_size,
        pin_mem=data["pin_memory"], num_workers=workers, world_size=world, rank=rank,
        seed=config["seed"], drop_last=True, channels=len(data["channel_names"]),
        num_patches=data["num_patches"], patch_samples=config["patch_encoder"]["patch_samples"])
    model = PretrainModel(config, device)
    if rank == 0:
        (output / "initialization.json").write_text(json.dumps(fingerprint(model), indent=2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=optimization["base_learning_rate"],
                                  betas=tuple(optimization["adam_betas"]), eps=optimization["adam_epsilon"],
                                  weight_decay=optimization["weight_decay"])
    microbatches = len(loader) // accumulation * accumulation
    updates = microbatches // accumulation
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=optimization["epochs"] * updates, eta_min=optimization["min_learning_rate"])
    start = 0
    step = 0
    # All ranks initialize identical weights first. Resume restores saved RNG below.
    initialize_pretrain_rng(config, rank)
    if args.resume:
        start, extra = load_checkpoint(ROOT / args.resume, model, optimizer, scheduler, device, rank,
                                       expected_masking=config["masking"],
                                       expected_pretrain_rng=pretrain_rng_mode(config))
        step = extra["step"]
    backbone = model.backbone
    decoder = model.decoder
    if args.distributed:
        backbone = DistributedDataParallel(backbone, device_ids=[device.index])
        decoder = DistributedDataParallel(decoder, device_ids=[device.index])
    rng_report = pretrain_rng_report(config, device, rank, bool(args.resume))
    rng_report.update(start_epoch=start, start_step=step)
    (output / ("rng-start-" + os.environ.get("SLURM_JOB_ID", "local") + "-rank" + str(rank) + ".json")).write_text(
        json.dumps(rng_report, indent=2))
    timer.start()
    for epoch in range(start, optimization["epochs"]):
        sampler.set_epoch(epoch)
        loader.generator.manual_seed(config["seed"] + epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for index, (signals, _) in enumerate(loader):
            if index == microbatches:
                break
            signals = signals.to(device, non_blocking=True) / data["value_scale"]
            update = (index + 1) % accumulation == 0
            diagnostic = update and (step == 0 or (step + 1) % config["runtime"]["log_every_steps"] == 0)
            capture(model, diagnostic)
            encoder_sync = nullcontext()
            decoder_sync = nullcontext()
            if args.distributed and not update:
                encoder_sync = backbone.no_sync()
                decoder_sync = decoder.no_sync()
            # Model.forward에도 같은 순서가 보인다. DDP에서는 두 모듈의 sync를 각각 제어한다.
            masks = make_masks(signals.shape[0], signals.shape[1], data["num_patches"],
                               config["masking"], device, model.mask_coordinates)
            coordinates = model.backbone.default_channel_coordinates[None].expand(signals.shape[0], -1, -1)
            valid = torch.ones(signals.shape[:2], dtype=torch.bool, device=device)
            with encoder_sync, decoder_sync:
                with autocast(device):
                    context = backbone(signals, coordinates, valid, masks["context_mask"])
                    with torch.no_grad():
                        target_grid = signals.unfold(-1, model.patch_samples, model.patch_samples)
                        target = gather_targets(target_grid, masks["target_blocks"])
                    prediction = decoder(context, masks)
                    loss = reconstruction_loss(prediction, target, masks["target_token_valid"],
                                               config["mae"]["smooth_l1_beta"])
                (loss / accumulation).backward()
            if update:
                details = {}
                if diagnostic:
                    details = measurements(model)
                    details.update(reconstruction(prediction, target))
                    for key, value in masks.get("masking_diagnostics", {}).items():
                        details["masking/" + key] = float(value)
                    details["masking/target_tokens"] = float(masks["target_mask"].sum((1, 2)).float().mean())
                    details["masking/context_tokens"] = float(masks["context_mask"].sum((1, 2)).float().mean())
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), optimization["gradient_clip_norm"],
                                                      error_if_nonfinite=True)
                learning_rate = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()  # GR9-1 pretrain은 optimizer 다음에 cosine을 진행한다.
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if rank == 0 and diagnostic:
                    append_record(output / "metrics.jsonl", {"epoch": epoch + 1, "step": step,
                                  "loss": float(loss), "lr_used": learning_rate, "pre_clip_norm": float(norm),
                                  "clipped": float(norm) > optimization["gradient_clip_norm"], **details})
                if args.smoke or timer.finished():
                    break
        states = collect_rng(device, world)
        if smoke_seconds:
            import resource
            import time
            smoke_resources = dict(elapsed_seconds=time.monotonic() - timer.started, rank=rank, steps=step,
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
                process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                children_max_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
            (output / ("smoke-resources-rank%d.json" % rank)).write_text(json.dumps(smoke_resources, indent=2))
        if rank == 0:
            extra = {"step": step, "dataset_fingerprint": dataset.fingerprint,
                     "partial_epoch_smoke": partial_smoke}
            save_checkpoint(output / "last.pth", model, optimizer, scheduler, epoch + 1, config, states, extra)
            if (epoch + 1) % config["runtime"]["save_every_epochs"] == 0 and not partial_smoke:
                path = output / f"checkpoint-epoch-{epoch + 1:04d}.pth"
                save_checkpoint(path, model, optimizer, scheduler, epoch + 1, config, states, extra)
                saved = sorted(output.glob("checkpoint-epoch-*.pth"))
                for obsolete in saved[:-config["runtime"]["keep_last_checkpoints"]]:
                    obsolete.unlink()
        if partial_smoke:
            break
    dataset.close()
    if args.distributed:
        dist.destroy_process_group()


def build_finetune(config):
    checkpoint = torch.load(ROOT / config["model"]["checkpoint"], map_location="cpu")
    pretrain = checkpoint["config"]
    # Raw encoder/PE ablations store an ablation wrapper in their checkpoint.
    # Recreate that exact backbone before strict loading; constructing the
    # default EEGEncoder would silently change the number of MJDE stages.
    if "ablation" in pretrain:
        from ablation.models import build_backbone
        backbone = build_backbone(pretrain)
    else:
        backbone = EEGEncoder(pretrain)
    expected_keys = set(backbone.state_dict())
    weights = {}
    for key, value in checkpoint["model"].items():
        if key.startswith("backbone."):
            key = key[len("backbone."):]
            # GR2 campaign snapshots used an adapter named ``encoder.core``
            # while the downstream EEGEncoder owns that module directly as
            # ``encoder``.  This is a namespace-only compatibility mapping;
            # strict loading below still verifies every tensor and shape.
            if key not in expected_keys and key.startswith("encoder.core."):
                candidate = "encoder." + key[len("encoder.core."):]
                if candidate in expected_keys:
                    key = candidate
            # Raw ablation checkpoints save the wrapped core as ``encoder``;
            # AblationBackbone exposes it as ``encoder.core`` at finetune.
            if key not in expected_keys and key.startswith("encoder."):
                candidate = "encoder.core." + key[len("encoder."):]
                if candidate in expected_keys:
                    key = candidate
            if key in weights:
                raise ValueError("Duplicate backbone checkpoint key after mapping: " + key)
            weights[key] = value
    backbone.load_state_dict(weights, strict=True)
    spec = get_dataset_spec(config["data"]["dataset"])
    patches = spec.signal_length // pretrain["patch_encoder"]["patch_samples"]
    head = config["model"]
    if config["data"]["dataset"] == "isruc":
        return SleepModel(backbone, spec.num_channels, patches, head["head_dropout"])
    hidden = head["head_hidden_tokens"]
    if hidden is None:
        hidden = patches
    return FinetuneModel(backbone, spec.num_channels, patches, spec.num_outputs, head["head_dropout"], hidden)


def optimizer_groups(model, config):
    encoder_parameters = list(model.backbone.position.parameters())
    encoder_parameters.extend(model.backbone.encoder.parameters())
    # A head-only control keeps the pretrained tokenizer/position/encoder
    # immutable.  Do not hand frozen tensors to AdamW: this makes the
    # optimizer audit unambiguous and prevents accidental decoupled decay.
    tokenizer_parameters = [item for item in model.backbone.tokenizer.parameters() if item.requires_grad]
    encoder_parameters = [item for item in encoder_parameters if item.requires_grad]
    head_parameters = [item for item in model.head.parameters() if item.requires_grad]
    groups = []
    if tokenizer_parameters:
        groups.append({"name": "tokenizer", "params": tokenizer_parameters, "lr": config["tokenizer_learning_rate"]})
    if encoder_parameters:
        groups.append({"name": "encoder", "params": encoder_parameters, "lr": config["encoder_learning_rate"]})
    if head_parameters:
        groups.append({"name": "head", "params": head_parameters, "lr": config["head_learning_rate"]})
    if not groups:
        raise ValueError("finetune has no trainable parameters")
    return groups


def predict(model, batch, device):
    return model(batch["x"].to(device, non_blocking=True),
                 batch["channel_coordinates"].to(device, non_blocking=True),
                 batch["channel_validity"].to(device, non_blocking=True))


@torch.no_grad()
def evaluate(model, loader, task, dataset_name, device, world):
    model.eval()
    predictions = []
    labels = []
    subjects = []
    cells = {}
    summary = None

    def record_summary(module, args, features):
        nonlocal summary
        if dataset_name == "isruc":
            summary = features.detach().cpu().reshape(-1, features.shape[-1])
        else:
            valid = args[2][:, :, None, None].expand(-1, -1, features.shape[2], 1)
            values = features * valid
            summary = (values.sum((1, 2)) / valid.sum((1, 2)).clamp_min(1)).detach().cpu()

    summary_module = model.backbone
    if dataset_name == "isruc":
        summary_module = model.head["sequence_encoder"]
    handle = summary_module.register_forward_hook(record_summary)
    for index, batch in enumerate(loader):
        capture(model, index == 0)
        with autocast(device):
            logits = predict(model, batch, device)
        truth = batch["label"]
        repeats = truth.numel() // truth.shape[0]
        if truth.ndim > 1:
            logits = logits.reshape(-1, logits.shape[-1])
        predictions.append(logits.detach().cpu())
        labels.append(truth.reshape(-1))
        batch_subjects = []
        for subject in batch["subject_id"]:
            batch_subjects.extend([person_id(dataset_name, subject)] * repeats)
        subjects.extend(batch_subjects)
        for embedding, label, person in zip(summary, truth.reshape(-1), batch_subjects):
            category = "continuous"
            if task != "regression":
                category = int(label)
            key = (person, category)
            if key not in cells:
                cells[key] = []
            if len(cells[key]) < 16:
                cells[key].append((embedding, float(label)))
    handle.remove()
    local = (torch.cat(predictions), torch.cat(labels), subjects, cells)
    shards = [local]
    if world > 1:
        shards = [None] * world
        dist.all_gather_object(shards, local)
    predictions, labels, subjects = [], [], []
    cells = {}
    for logits, truth, people, shard_cells in shards:
        predictions.append(logits)
        labels.append(truth)
        subjects.extend(people)
        for key, values in shard_cells.items():
            if key not in cells:
                cells[key] = []
            cells[key].extend(values)
    logits = torch.cat(predictions)
    truth = torch.cat(labels)
    result = metrics(task, logits, truth)
    result["diagnostics"] = prediction_report(task, logits, truth, subjects, dataset_name)
    result["diagnostics"]["probe"] = probe_report(cells, task)
    return result


def run_finetune(config, args, policy=None):
    device, rank, world = setup(args.device, args.distributed, config["seed"], False, False, True)
    data = config["data"]
    optimization = config["optimization"]
    spec = get_dataset_spec(data["dataset"])
    output = ROOT / config["runtime"]["output"]
    if args.smoke:
        output = ROOT / "outputs/smoke" / data["dataset"]
    output.mkdir(parents=True, exist_ok=True)
    batch_size = optimization["batch_size_per_gpu"]
    workers = data["num_workers"]
    accumulation = optimization["gradient_accumulation_steps"]
    smoke_batches = getattr(args, "smoke_batches", 0)
    partial_smoke = args.smoke or bool(smoke_batches)
    if args.smoke:
        batch_size, workers, accumulation = 2, 0, 1
    datasets = {}
    loaders = {}
    for split in ("train", "val", "test"):
        datasets[split] = spec.dataset_class(data["dataset_dir"], split)
        datasets[split].enable_coordinate_only_channels()
        if split == "train":
            sampler = DistributedSampler(datasets[split], num_replicas=world, rank=rank,
                                          shuffle=True, seed=config["seed"])
        else:
            # 원본 ExactDistributedSampler와 같은 간격으로 평가 데이터를 나눈다.
            sampler = range(rank, len(datasets[split]), world)
            if smoke_batches:
                # Spread evaluation across the split instead of reading only
                # the first subject/class. Diagnostic metrics are not results.
                count = min(len(datasets[split]), smoke_batches * batch_size)
                sampler = torch.linspace(0, len(datasets[split]) - 1, count).long().tolist()
        loaders[split] = DataLoader(datasets[split], batch_size=batch_size, sampler=sampler,
                                    num_workers=workers, pin_memory=True)
        if smoke_batches and split == "train":
            loaders[split] = LimitedLoader(loaders[split], smoke_batches * accumulation)
    model = build_finetune(config).to(device)
    if config["model"].get("freeze_backbone", False):
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
        if not any(parameter.requires_grad for parameter in model.head.parameters()):
            raise ValueError("freeze_backbone requires a trainable task head")
    if rank == 0:
        (output / "initialization.json").write_text(json.dumps(fingerprint(model), indent=2))
    optimizer = torch.optim.AdamW(optimizer_groups(model, optimization),
                                  betas=tuple(optimization["adam_betas"]), eps=optimization["adam_epsilon"],
                                  weight_decay=optimization["weight_decay"])
    updates = math.ceil(len(loaders["train"]) / accumulation)
    scheduler_factory = GroupCosineScheduler if policy is None else policy.make_scheduler
    scheduler = scheduler_factory(optimizer, optimization["epochs"] * updates,
                                     optimization["min_learning_rate"])
    selectors = ["balanced_accuracy", "kappa"]
    if spec.task == "binary":
        selectors = ["balanced_accuracy", "auroc"]
    if spec.task == "regression":
        selectors = ["r2"]
    best = {}
    for name in selectors:
        best[name] = {"score": -float("inf"), "epoch": 0}
    start, step = 0, 0
    if args.resume:
        start, extra = load_checkpoint(ROOT / args.resume, model, optimizer, scheduler, device, rank)
        step, best = extra["step"], extra["best"]
    trainer = model
    if args.distributed:
        trainer = DistributedDataParallel(model, device_ids=[device.index],
                                          find_unused_parameters=bool(policy and policy.head_first_epochs))
    for epoch in range(start, optimization["epochs"]):
        loaders["train"].sampler.set_epoch(epoch)
        trainer.train()
        if policy is not None:
            policy.on_epoch_start(model, epoch)
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(loaders["train"]):
            group_start = index // accumulation * accumulation
            group_size = min(accumulation, len(loaders["train"]) - group_start)
            update = (index + 1) % accumulation == 0 or index + 1 == len(loaders["train"])
            diagnostic = update and (step == 0 or (step + 1) % config["runtime"]["log_every_steps"] == 0)
            capture(model, diagnostic)
            labels = batch["label"].to(device, non_blocking=True)
            with autocast(device):
                logits = predict(trainer, batch, device)
                loss = downstream_loss(spec.task, logits, labels, optimization)
            (loss / group_size).backward()
            if update:
                details = measurements(model)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), optimization["gradient_clip_norm"],
                                                      error_if_nonfinite=True)
                learning_rates = scheduler.step()  # Downstream은 optimizer 직전에 LR을 설정한다.
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if rank == 0 and diagnostic:
                    append_record(output / "metrics.jsonl", {"epoch": epoch + 1, "step": step,
                                  "loss": float(loss), "lr_used": learning_rates,
                                  "pre_clip_norm": float(norm), "clipped": float(norm) > optimization["gradient_clip_norm"],
                                  **details})
                if args.smoke:
                    break
        if not args.smoke:
            validation = evaluate(model, loaders["val"], spec.task, data["dataset"], device, world)
            if rank == 0:
                append_record(output / "validation.jsonl", {"epoch": epoch + 1, **validation})
            for selector in selectors:
                score = validation[selector]
                if best[selector]["epoch"] == 0 or score > best[selector]["score"]:
                    best[selector] = {"score": score, "epoch": epoch + 1}
                    if rank == 0:
                        torch.save(model.state_dict(), output / ("best-" + selector + ".pth"))
        states = collect_rng(device, world)
        if rank == 0:
            save_checkpoint(output / "last.pth", model, optimizer, scheduler, epoch + 1, config, states,
                            {"step": step, "best": best, "partial_epoch_smoke": partial_smoke})
        if partial_smoke:
            break
    if args.distributed:
        dist.barrier()
    if not args.smoke:
        result = {}
        for selector in selectors:
            model.load_state_dict(torch.load(output / ("best-" + selector + ".pth"), map_location=device), strict=True)
            test = evaluate(model, loaders["test"], spec.task, data["dataset"], device, world)
            result[selector] = {"selection": best[selector], "test": test}
        if rank == 0:
            filename = "smoke_result.json" if smoke_batches else "result.json"
            (output / filename).write_text(json.dumps(result, indent=2))
    if args.distributed:
        dist.destroy_process_group()
