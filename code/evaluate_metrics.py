"""Metric computation for ID vs OOD discrimination."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import auc, roc_curve, precision_recall_curve


@dataclass
class OODMetrics:
    auroc: float
    detection_error: float
    aupr_in: float
    aupr_out: float
    threshold: float


def compute_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return auc(fpr, tpr), fpr, tpr, thresholds


def compute_detection_error(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray, pos_ratio: float = 0.5) -> tuple[float, float]:
    fnr = 1.0 - tpr
    errors = pos_ratio * fnr + (1.0 - pos_ratio) * fpr
    idx = int(np.argmin(errors))
    return float(errors[idx]), float(thresholds[idx])


def compute_aupr(id_scores: np.ndarray, ood_scores: np.ndarray) -> tuple[float, float]:
    y_true_in = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])
    precision_in, recall_in, _ = precision_recall_curve(y_true_in, y_score)
    aupr_in = auc(recall_in, precision_in)

    y_true_out = 1 - y_true_in
    precision_out, recall_out, _ = precision_recall_curve(y_true_out, -y_score)
    aupr_out = auc(recall_out, precision_out)
    return float(aupr_in), float(aupr_out)


def evaluate_ood(id_scores: np.ndarray, ood_scores: np.ndarray) -> OODMetrics:
    auroc, fpr, tpr, thresholds = compute_auroc(id_scores, ood_scores)
    detection_error, threshold = compute_detection_error(fpr, tpr, thresholds)
    aupr_in, aupr_out = compute_aupr(id_scores, ood_scores)
    return OODMetrics(
        auroc=float(auroc),
        detection_error=float(detection_error),
        aupr_in=aupr_in,
        aupr_out=aupr_out,
        threshold=threshold,
    )


if __name__ == '__main__':
    rng = np.random.default_rng(42)
    id_scores = rng.normal(loc=0.85, scale=0.08, size=300)
    ood_scores = rng.normal(loc=0.45, scale=0.12, size=300)
    metrics = evaluate_ood(id_scores, ood_scores)
    print(metrics)
