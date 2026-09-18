"""Frozen datasets and task models; output only small real samples and comparisons."""

import gc
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch.utils.data import default_collate
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.datasets.registry import get_dataset_spec
from src.models.eeg_encoder import build_eeg_encoder
from src.models.finetune.task_model import TaskModel
from src.models.finetune.model_for_isruc import Model as SleepModel
from src.utils.classification_losses import classification_loss

torch.set_num_threads(2)
root = Path(__file__).resolve().parents[2]
checkpoint = torch.load(root / "tests/reference/checkpoint-epoch-0040.pth", map_location="cpu")
pretrain = checkpoint["resolved_config"]
roster = ["tuab", "tuev", "chb", "seedv", "faced",
          "mentalarithmetic", "bciciv2a", "physionet_mi", "isruc", "hmc", "siena"]
report = {}
for task_name in roster:
    path = root / "configs/downstream" / ("gr9-1_" + task_name + "_seed42.yaml")
    config = yaml.safe_load(path.read_text())
    dataset_name = config["data"]["dataset"]
    spec = get_dataset_spec(dataset_name)
    item = {"splits": {}}
    for split in ("train", "val", "test"):
        dataset = spec.dataset_class(config["data"]["dataset_dir"], split)
        dataset.enable_coordinate_only_channels()
        indices = [0, len(dataset) - 1]
        samples = [dataset[index] for index in indices]
        item["splits"][split] = {"count": len(dataset), "samples": samples}
        del dataset
        gc.collect()
    samples = item["splits"]["train"]["samples"]
    if dataset_name == "isruc":
        samples = samples[:1]
    batch = default_collate(samples)
    torch.manual_seed(721)
    encoder = build_eeg_encoder(pretrain)
    encoder.load_state_dict(checkpoint["context_encoder"], strict=True)
    patches = spec.signal_length // 200
    if dataset_name == "isruc":
        model = SleepModel(encoder, 200, spec.num_channels, patches,
                           config["model"]["head_dropout"], "all_patch_reps")
    else:
        model = TaskModel(encoder, 200, spec.num_outputs, config["model"]["head_dropout"],
                          "all_patch_reps", spec.num_channels, patches,
                          head_hidden_tokens=config["model"]["head_hidden_tokens"])
    hashes = {}
    for key, value in model.head.state_dict().items():
        hashes[key] = hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
    item["head_hashes"] = hashes
    model.eval()
    with torch.no_grad():
        item["logits"] = model(batch["x"], batch["channel_coordinates"], batch["channel_region_ids"], batch["channel_validity"])
    if dataset_name in ("seed-v", "stress"):
        model.train()
        torch.manual_seed(771)
        item["rng_before_train"] = torch.get_rng_state()
        logits = model(batch["x"], batch["channel_coordinates"], batch["channel_region_ids"], batch["channel_validity"])
        if spec.task == "regression":
            loss = torch.nn.functional.mse_loss(logits.reshape(-1), batch["label"].to(logits.dtype).reshape(-1))
        else:
            loss_config = dict(config["optimization"])
            loss_config["classification_loss"] = "weighted_ce" if spec.task == "binary" else "ce"
            loss = classification_loss(spec.task, logits, batch["label"], loss_config)
        loss.backward()
        item["train_logits"] = logits.detach()
        item["loss"] = loss.detach()
        item["rng_after_train"] = torch.get_rng_state()
        item["gradients"] = {}
        for key, parameter in model.named_parameters():
            item["gradients"][key] = parameter.grad.detach().clone()
    torch.save(item, root / "outputs" / ("oracle-" + dataset_name + ".pth"))
    report[dataset_name] = {split: item["splits"][split]["count"] for split in item["splits"]}
    print(dataset_name, report[dataset_name], flush=True)
    del model, encoder, item
    gc.collect()
(root / "outputs/oracle-datasets.json").write_text(json.dumps(report, indent=2))
