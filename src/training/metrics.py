"""원본과 같은 threshold 0.5, PR 곡선 사다리꼴 면적, BAcc/Kappa/R2."""

import numpy as np
from sklearn.metrics import (auc, balanced_accuracy_score, cohen_kappa_score,
                             f1_score, mean_squared_error, precision_recall_curve,
                             r2_score, roc_auc_score)


def metrics(task, logits, labels):
    truth = labels.numpy()
    if task == "regression":
        prediction = logits.float().numpy()
        return {"pearson": float(np.corrcoef(truth, prediction)[0, 1]),
                "r2": float(r2_score(truth, prediction)),
                "rmse": float(mean_squared_error(truth, prediction, squared=False))}
    if task == "binary":
        probability = logits.float().sigmoid().numpy()
        prediction = (probability >= 0.5).astype(np.int64)
        precision, recall, _ = precision_recall_curve(truth, probability, pos_label=1)
        return {"balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
                "auprc": float(auc(recall, precision)),
                "auroc": float(roc_auc_score(truth, probability))}
    prediction = logits.argmax(dim=-1).numpy()
    return {"balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "kappa": float(cohen_kappa_score(truth, prediction)),
            "weighted_f1": float(f1_score(truth, prediction, average="weighted"))}
