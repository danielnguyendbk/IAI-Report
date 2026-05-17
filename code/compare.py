"""Compare Baseline, Deep Ensemble, and TC outputs.

Expected input format:
Each model writes a CSV with a single column named `score`.
Higher score means the sample looks more like ID.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass
from sklearn.metrics import roc_curve

from evaluate_metrics import evaluate_ood


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / 'results'
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


def plot_simple_density(name: str, id_csv: str | Path, ood_csv: str | Path) -> None:
    id_scores = load_scores(id_csv)
    ood_scores = load_scores(ood_csv)
    
    plt.figure(figsize=(6, 4))
    
    # Tinh toan mean va std de dung duong cong qua chuong (Gaussian Bell Curve)
    mean_id, std_id = np.mean(id_scores), np.std(id_scores)
    mean_ood, std_ood = np.mean(ood_scores), np.std(ood_scores)
    
    # Tao dai gia tri Confidence Score tu 0 den 1
    x = np.linspace(0.0, 1.0, 300)
    
    # Cong thuc PDF cua phan phoi Gaussian chuan
    pdf_id = (1.0 / (std_id * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_id) / std_id) ** 2)
    pdf_ood = (1.0 / (std_ood * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_ood) / std_ood) ** 2)
    
    # Ve cac duong cong tron tru de dang ve tay tren giay
    plt.plot(x, pdf_id, label='Dữ liệu sạch (ID)', color='tab:blue', linewidth=2.5)
    plt.plot(x, pdf_ood, label='Dữ liệu nhiễu (OOD)', color='tab:orange', linewidth=2.5)
    
    # Ve mot duong xich-ma cat giua lam nguong phan loai lam sang
    threshold = (mean_id + mean_ood) / 2
    plt.axvline(threshold, color='red', linestyle='--', label=f'Ngưỡng tối ưu ({threshold:.3f})')
    
    plt.xlabel('Điểm độ tin cậy (Confidence Score)')
    plt.ylabel('Mật độ phân phối (Density)')
    plt.title(f'Phân phối quả chuông đơn giản (Dễ vẽ tay) - {name}')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'simple_density_{name.lower().replace(" ", "_")}.png', dpi=200)
    plt.close()


def main() -> None:
    model_files = {
        'Baseline': (RESULT_DIR / 'baseline_id.csv', RESULT_DIR / 'baseline_ood.csv'),
        'Deep Ensemble': (RESULT_DIR / 'de_id.csv', RESULT_DIR / 'de_ood.csv'),
        'TC': (RESULT_DIR / 'tc_id.csv', RESULT_DIR / 'tc_ood.csv'),
    }

    rows = []
    for model_name, (id_csv, ood_csv) in model_files.items():
        rows.append(evaluate_model(model_name, id_csv, ood_csv))
        plot_histogram(model_name, id_csv, ood_csv)
        plot_simple_density(model_name, id_csv, ood_csv)

    df = pd.DataFrame(rows)
    RESULT_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULT_DIR / 'metrics_summary.csv', index=False)
    plot_roc_curves(model_files)
    
    # In ra bang so sanh theo dinh dang Markdown/Text dep mat
    print("\n" + "="*85)
    print(" BẢNG TỔNG HỢP KẾT QUẢ SO SÁNH CÁC MÔ HÌNH (OOD DETECTION)")
    print("="*85)
    print(f"| {'Mô hình':<15} | {'AUROC (%)':<12} | {'Detection Error (%)':<19} | {'AUPR-In':<10} | {'AUPR-Out':<10} |")
    print(f"|{'-'*17}|{'-'*14}|{'-'*21}|{'-'*12}|{'-'*12}|")
    for row in rows:
        print(f"| {row['model']:<15} | {row['auroc']*100:>10.2f} % | {row['detection_error']*100:>17.2f} % | {row['aupr_in']:>10.4f} | {row['aupr_out']:>10.4f} |")
    print("="*85 + "\n")

    # In ra danh sach cac file da luu nhu file train
    print("="*85)
    print(" DANH SÁCH FILE KẾT QUẢ ĐÃ ĐƯỢC SINH RA & LƯU TRỮ:")
    print("="*85)
    print(f"  [BẢNG EXCEL]  {RESULT_DIR / 'metrics_summary.csv'}")
    print(f"  [ĐỒ THỊ ROC]  {FIG_DIR / 'roc_comparison.png'}")
    print(f"  [ĐỒ THỊ CỘT]  {FIG_DIR / 'hist_baseline.png'}")
    print(f"                {FIG_DIR / 'hist_deep_ensemble.png'}")
    print(f"                {FIG_DIR / 'hist_tc.png'}")
    print(f"  [VẼ TAY GIẤY] {FIG_DIR / 'simple_density_baseline.png'}")
    print(f"                {FIG_DIR / 'simple_density_deep_ensemble.png'}")
    print(f"                {FIG_DIR / 'simple_density_tc.png'}")
    print("="*85 + "\n")


if __name__ == '__main__':
    main()
