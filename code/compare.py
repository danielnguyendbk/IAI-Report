"""Compare Baseline, Deep Ensemble, and TC outputs.

Expected input format:
Each model writes a CSV with a single column named `score`.
Higher score means the sample looks more like ID.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from evaluate_metrics import evaluate_ood


RESULT_DIR = Path('results')
FIG_DIR = RESULT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_scores(csv_path: str | Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if 'score' not in df.columns:
        raise ValueError(f'{csv_path} must contain a score column')
    return df['score'].to_numpy(dtype=float)


def evaluate_model(name: str, id_csv: str | Path, ood_csv: str | Path) -> Dict[str, float]:
    id_scores = load_scores(id_csv)
    ood_scores = load_scores(ood_csv)
    metrics = evaluate_ood(id_scores, ood_scores)
    return {
        'model': name,
        'auroc': metrics.auroc,
        'detection_error': metrics.detection_error,
        'aupr_in': metrics.aupr_in,
        'aupr_out': metrics.aupr_out,
        'threshold': metrics.threshold,
    }


def plot_roc_curves(model_files: Dict[str, tuple[str, str]]) -> None:
    plt.figure(figsize=(7, 5))
    for model_name, (id_csv, ood_csv) in model_files.items():
        id_scores = load_scores(id_csv)
        ood_scores = load_scores(ood_csv)
        y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
        y_score = np.concatenate([id_scores, ood_scores])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, label=model_name)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Comparison for OOD Detection')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'roc_comparison.png', dpi=200)
    plt.close()


def plot_histogram(name: str, id_csv: str | Path, ood_csv: str | Path) -> None:
    id_scores = load_scores(id_csv)
    ood_scores = load_scores(ood_csv)
    plt.figure(figsize=(7, 5))
    plt.hist(id_scores, bins=30, alpha=0.6, label='ID')
    plt.hist(ood_scores, bins=30, alpha=0.6, label='OOD')
    plt.xlabel('Confidence / OOD Score')
    plt.ylabel('Count')
    plt.title(f'Score Distribution - {name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'hist_{name.lower().replace(" ", "_")}.png', dpi=200)
    plt.close()


def main() -> None:
    model_files = {
        'Baseline': ('results/baseline_id.csv', 'results/baseline_ood.csv'),
        'Deep Ensemble': ('results/de_id.csv', 'results/de_ood.csv'),
        'TC': ('results/tc_id.csv', 'results/tc_ood.csv'),
    }

    rows = []
    for model_name, (id_csv, ood_csv) in model_files.items():
        rows.append(evaluate_model(model_name, id_csv, ood_csv))
        plot_histogram(model_name, id_csv, ood_csv)

    df = pd.DataFrame(rows)
    RESULT_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULT_DIR / 'metrics_summary.csv', index=False)
    plot_roc_curves(model_files)
    print(df)


if __name__ == '__main__':
    main()
