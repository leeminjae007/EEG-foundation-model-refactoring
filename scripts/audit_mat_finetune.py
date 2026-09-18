"""Read-only CPU replay of historical and current MAT fine-tuning checkpoints."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate
import yaml
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, precision_recall_curve, auc


def encoder_key(key):
    if key.startswith("patch_encoder."):
        return key.replace("patch_encoder.", "tokenizer.", 1)
    if key == "positional_encoder.spatial.projection.weight":
        return "position.projection.weight"
    if key == "positional_encoder.post_fusion_norm.weight":
        return "position.norm.weight"
    if key.startswith("context_encoder."):
        return key.replace("context_encoder.", "encoder.", 1).replace(".block.", ".").replace(".mlp.layers.", ".mlp.")
    return key


def mapped(key):
    if key.startswith("classifier.encoder."):
        return "backbone." + encoder_key(key[len("classifier.encoder."):])
    return key.replace("classifier.head.", "head.", 1)


def stats(logits, labels, threshold=0.5):
    probability = logits.float().sigmoid().numpy()
    y = labels.numpy()
    prediction = probability >= threshold
    cm = confusion_matrix(y, prediction, labels=[0, 1])
    precision, recall, _ = precision_recall_curve(y, probability)
    return dict(balanced_accuracy=float(balanced_accuracy_score(y, prediction)),
                auroc=float(roc_auc_score(y, probability)), auprc=float(auc(recall, precision)),
                confusion=cm.tolist(), recall=(cm.diagonal() / cm.sum(1)).tolist(),
                positive_fraction=float(prediction.mean()), threshold=float(threshold),
                probability_by_class={str(k):dict(mean=float(probability[y == k].mean()),
                    quantiles=np.quantile(probability[y == k], [.1, .5, .9]).tolist()) for k in (0, 1)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=("old", "ported", "current"), required=True)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--audit", required=True, type=Path)
    args = p.parse_args()
    torch.set_num_threads(4)
    args.audit.mkdir(parents=True, exist_ok=True)
    original = args.root.parent / "eeg-foundation-model"
    campaign = args.root / "outputs/knn37_optuna_valselect_20260917_0205"
    old_outputs = original / "outputs/gr9_1_downstream_20260910/gr9_1_equal_shpe_gelu_rmsnorm_d200_e40_20260910/stress/eeg_mae/all_patch_reps"
    sys.path.insert(0, str(original if args.variant == "old" else campaign / "source"))
    if args.variant == "old":
        from src.datasets.stress_dataset import StressDataset
        get_dataset_spec = lambda name: SimpleNamespace(dataset_class=StressDataset)
        from src.models.eeg_encoder import build_eeg_encoder
        from src.models.finetune.task_model import TaskModel
        from src.utils.classification_losses import classification_loss
        from src.utils.metrics import chb_metrics as metric_function
    else:
        from src.data.datasets.registry import get_dataset_spec
        from src.training.engine import build_finetune
        from src.modules.loss import downstream_loss
        from src.training.metrics import metrics
        metric_function = lambda logits, labels: metrics("binary", logits, labels)
    spec = get_dataset_spec("stress")
    cfg = yaml.safe_load((campaign / "datasets/mentalarithmetic/trial-000/configs/42.yaml").read_text())
    datasets = {}
    fingerprints = {}
    for split in ("train", "val", "test"):
        dataset = spec.dataset_class(cfg["data"]["dataset_dir"], split)
        dataset.enable_coordinate_only_channels()
        datasets[split] = dataset
        h = hashlib.sha256()
        counts = [0, 0]
        for sample in dataset:
            for name in sorted(sample):
                value = sample[name]
                h.update(name.encode())
                h.update(value.numpy().tobytes() if torch.is_tensor(value) else str(value).encode())
            counts[int(sample["label"])] += 1
        fingerprints[split] = dict(count=len(dataset), class_counts=counts, sha256=h.hexdigest(),
                                   active_channels=int(dataset.channel_validity.sum()))
    report = dict(variant=args.variant, device="cpu", precision="float32", datasets=fingerprints, seeds={})

    def forward(model, batch):
        if args.variant == "old":
            return model(batch["x"], batch["channel_coordinates"], batch["channel_region_ids"], batch["channel_validity"])
        return model(batch["x"], batch["channel_coordinates"], batch["channel_validity"])

    for seed in (42, 696, 1001, 1234, 3407):
        torch.manual_seed(seed)
        current_dir = campaign / "datasets/mentalarithmetic/trial-000" / ("seed" + str(seed))
        current_cfg = yaml.safe_load((current_dir / "resolved_config.yaml").read_text())
        historical_dir = old_outputs / ("seed-" + str(seed))
        old_cfg = yaml.safe_load((historical_dir / "resolved_config.yaml").read_text())
        if args.variant == "old":
            raw = torch.load(old_cfg["model"]["checkpoint"], map_location="cpu")
            encoder = build_eeg_encoder(raw["resolved_config"])
            model = TaskModel(encoder, 200, 1, .2, "all_patch_reps", 20, 5)
            saved = torch.load(historical_dir / "best-balanced_accuracy.pth", map_location="cpu")
            model.load_state_dict(saved["model"], strict=True)
            del raw, saved
        else:
            if args.variant == "ported":
                current_cfg["model"]["checkpoint"] = str(args.root / "outputs/gr9_1_epoch40.pth")
            model = build_finetune(current_cfg)
            path = (historical_dir if args.variant == "ported" else current_dir) / "best-balanced_accuracy.pth"
            saved = torch.load(path, map_location="cpu")
            state = {mapped(k):v for k,v in saved["model"].items()} if args.variant == "ported" else saved
            model.load_state_dict(state, strict=True)
            del saved, state
        model.eval()
        outputs = {}
        item = {}
        for split in ("val", "test"):
            logits, labels = [], []
            with torch.no_grad():
                for batch in DataLoader(datasets[split], batch_size=16, shuffle=False, num_workers=0):
                    logits.append(forward(model, batch).detach().cpu())
                    labels.append(batch["label"])
            x, y = torch.cat(logits), torch.cat(labels)
            outputs[split] = dict(logits=x, labels=y)
            item[split] = stats(x, y)
            item[split]["implementation_metrics"] = metric_function(x, y)
            if args.variant == "ported":
                reference = torch.load(args.audit / ("old-seed%d.pth" % seed), map_location="cpu")[split]
                assert torch.equal(y, reference["labels"])
                item[split]["old_max_logit_abs"] = float((x - reference["logits"]).abs().max())
                item[split]["old_prediction_disagreements"] = int(((x >= 0) != (reference["logits"] >= 0)).sum())
        # Fit a diagnostic threshold on validation only, never on test.
        val_x, val_y = outputs["val"]["logits"], outputs["val"]["labels"]
        candidates = np.unique(np.r_[0., val_x.sigmoid().numpy(), 1.])
        threshold = max(candidates, key=lambda t: (balanced_accuracy_score(val_y.numpy(), val_x.sigmoid().numpy() >= t), -abs(float(t) - .5)))
        item["validation_threshold_diagnostic"] = dict(threshold=float(threshold),
            validation=stats(val_x, val_y, threshold), test=stats(outputs["test"]["logits"], outputs["test"]["labels"], threshold))
        torch.save(outputs, args.audit / (args.variant + "-seed%d.pth" % seed))
        if seed == 42 and args.variant in ("old", "ported"):
            dataset = datasets["train"]
            batch = default_collate([dataset[i] for i in [0, 1, len(dataset)-2, len(dataset)-1]])
            model.train()
            model.zero_grad(set_to_none=True)
            torch.manual_seed(771)
            logits = forward(model, batch)
            loss = classification_loss("binary", logits, batch["label"], old_cfg["optimization"]) if args.variant == "old" else downstream_loss("binary", logits, batch["label"], current_cfg["optimization"])
            loss.backward()
            gradients = {(mapped(k) if args.variant == "old" else k):v.grad.detach().clone() for k,v in model.named_parameters() if v.grad is not None}
            if args.variant == "old":
                torch.save(dict(logits=logits.detach(), loss=loss.detach(), gradients=gradients), args.audit / "old-gradient.pth")
            else:
                reference = torch.load(args.audit / "old-gradient.pth", map_location="cpu")
                assert gradients.keys() == reference["gradients"].keys()
                item["gradient_equivalence"] = dict(loss_abs=float((loss-reference["loss"]).abs()),
                    logits_max_abs=float((logits-reference["logits"]).abs().max()),
                    gradient_max_abs=max(float((v-reference["gradients"][k]).abs().max()) for k,v in gradients.items()),
                    tensors=len(gradients))
                del reference
            del gradients, logits, loss
        report["seeds"][str(seed)] = item
        (args.audit / (args.variant + "-report.json")).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(dict(variant=args.variant, seed=seed, test=item["test"], equivalence=item.get("gradient_equivalence"))), flush=True)
        del model
        gc.collect()


if __name__ == "__main__":
    main()
