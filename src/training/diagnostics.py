"""학습 난수와 gradient를 바꾸지 않는 측정값."""

import hashlib
import re
import numpy as np
import torch
from sklearn.metrics import (auc, balanced_accuracy_score, confusion_matrix, precision_recall_curve,
                             recall_score, roc_auc_score)
from src.modules.attention import MaskedAttention, DecoderAttention
from src.modules.position_embedding import PositionEmbedding
from src.modules.fusion import PatchFusionGate


def capture(model, enabled):
    for module in model.modules():
        if isinstance(module, (MaskedAttention, DecoderAttention, PositionEmbedding, PatchFusionGate)):
            module.capture = enabled
            module.diagnostics = {}


def measurements(model):
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, (MaskedAttention, DecoderAttention, PositionEmbedding, PatchFusionGate)):
            for key, value in module.diagnostics.items():
                result[name + "/" + key] = float(value)
        if isinstance(module, PatchFusionGate):
            for parameter_name, parameter in module.named_parameters(recurse=False):
                if parameter.grad is not None:
                    result[name + "/gate/" + parameter_name + "_gradient_rms_pre_clip"] = float(
                        parameter.grad.detach().float().square().mean().sqrt())
    for name, parameter in model.named_parameters():
        if name.endswith("fusion_gates"):
            values = parameter.detach().float()
            result[name + "/value_mean"] = float(values.mean())
            result[name + "/sigmoid_mean"] = float(values.sigmoid().mean())
            if parameter.grad is not None:
                result[name + "/gradient_rms_pre_clip"] = float(parameter.grad.float().square().mean().sqrt())
    return result


def fingerprint(model):
    report = {"comparison_available": False,
              "reason": "Historical step-zero tensors were not saved; see separate equivalence replay.",
              "tensors": {}}
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        payload = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        report["tensors"][name] = {"shape": list(value.shape), "dtype": str(value.dtype),
                                    "sha256": hashlib.sha256(payload).hexdigest()}
    return report


@torch.no_grad()
def reconstruction(prediction, target):
    predicted = prediction.float().flatten()
    truth = target.float().flatten()
    centered_prediction = predicted - predicted.mean()
    centered_truth = truth - truth.mean()
    denominator = centered_prediction.norm() * centered_truth.norm()
    return {"reconstruction_correlation": float((centered_prediction * centered_truth).sum() / denominator.clamp_min(1e-12)),
            "reconstruction_prediction_rms": float(predicted.square().mean().sqrt()),
            "reconstruction_target_rms": float(truth.square().mean().sqrt())}


def person_id(dataset, value):
    person = str(value)
    if dataset == "stress":
        person = re.sub(r"^(Subject\d+)_[12]$", r"\1", person)
    return person


def prediction_report(task, logits, labels, subjects, dataset):
    if task == "regression":
        return {}
    truth = labels.long().numpy()
    if task == "binary":
        probability = logits.float().sigmoid().numpy()
        predicted = (probability >= 0.5).astype(np.int64)
        classes = 2
    else:
        probability = logits.float().softmax(-1).numpy()
        predicted = probability.argmax(-1)
        classes = logits.shape[-1]
    report = {"class_recall": recall_score(truth, predicted, labels=list(range(classes)),
                                           average=None, zero_division=0).tolist(),
              "subjects": {}, "missing_subject_samples": 0}
    cells = {}
    for index, value in enumerate(subjects):
        person = person_id(dataset, value)
        if person.lower() in ("unknown", "unavailable", "none", ""):
            report["missing_subject_samples"] += 1
            continue
        if person not in cells:
            cells[person] = []
        cells[person].append(index)
    for person, indices in cells.items():
        y = truth[indices]
        p = predicted[indices]
        matrix = confusion_matrix(y, p, labels=list(range(classes)))
        item = {"samples": len(indices), "fp_by_class": (matrix.sum(0) - matrix.diagonal()).tolist(),
                "fn_by_class": (matrix.sum(1) - matrix.diagonal()).tolist(),
                "ranking_eligible": bool(len(np.unique(y)) > 1)}
        if task == "binary" and item["ranking_eligible"]:
            precision, recall, _ = precision_recall_curve(y, probability[indices])
            item["auprc"] = float(auc(recall, precision))
            item["auroc"] = float(roc_auc_score(y, probability[indices]))
        report["subjects"][person] = item
    return report


def centroid_probe(embeddings, categories):
    """원본처럼 각 category를 절반으로 나누어 centroid 적합/평가한다."""
    categories = np.asarray(categories).astype(str)
    generator = np.random.default_rng(42)
    training = []
    validation = []
    retained = []
    for category in sorted(np.unique(categories)):
        indices = np.flatnonzero(categories == category)
        if indices.size < 2:
            continue
        indices = generator.permutation(indices)
        split = max(1, indices.size // 2)
        training.extend(indices[:split].tolist())
        validation.extend(indices[split:].tolist())
        retained.append(category)
    if len(retained) < 2:
        return {"defined": False, "num_classes": len(retained)}
    train = embeddings[training].float()
    test = embeddings[validation].float()
    mean = train.mean(dim=0, keepdim=True)
    scale = train.std(unbiased=False).clamp_min(1e-6)
    train = (train - mean) / scale
    test = (test - mean) / scale
    centroids = []
    for category in retained:
        centroids.append(train[categories[training] == category].mean(dim=0))
    distances = torch.cdist(test, torch.stack(centroids))
    prediction = np.asarray(retained)[distances.argmin(dim=1).numpy()]
    score = float(balanced_accuracy_score(categories[validation], prediction))
    chance = 1.0 / len(retained)
    return {"defined": True, "num_classes": len(retained), "balanced_accuracy": score,
            "chance": chance, "chance_normalized_accuracy": (score - chance) / (1.0 - chance)}


def probe_report(cells, task):
    embeddings, people, labels = [], [], []
    for key in sorted(cells):
        for embedding, label in cells[key][:16]:
            embeddings.append(embedding)
            people.append(key[0])
            labels.append(label)
    embeddings = torch.stack(embeddings)
    labels = torch.tensor(labels)
    if task == "regression":
        boundaries = torch.quantile(labels, torch.tensor([0.2, 0.4, 0.6, 0.8]))
        categories = torch.bucketize(labels, boundaries).tolist()
    else:
        categories = labels.long().tolist()
    known = []
    known_people = []
    for index, person in enumerate(people):
        if person.lower() not in ("unknown", "unavailable", "none", ""):
            known.append(index)
            known_people.append(person)
    subject = {"defined": False, "num_classes": 0}
    if known:
        subject = centroid_probe(embeddings[known], known_people)
    return {"person_class_stratified": task != "regression", "cells": len(cells),
            "samples": len(embeddings), "subject": subject,
            "label": centroid_probe(embeddings, categories)}
