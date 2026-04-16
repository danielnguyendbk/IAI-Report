"""
train_tc_kaggle.py — Script Train TC cho Kaggle (Phan C - An)
==============================================================
Dung du lieu Arrhythmia da lam sach tu repo nhom (nhanh daniel).
Xuat ra:
  1. Model .pt (toan bo chain)
  2. Score CSV (cho Thái chay compare.py)
  3. JSON predictions (% confidence + nhan lam sang)
  4. Bieu do + bang ket qua

Cach chay tren Kaggle:
  !git clone -b daniel https://github.com/danielnguyendbk/IAI-Report.git
  !git clone https://github.com/Lawliet-zzl/TC.git
  !cp IAI-Report/code/data/Arrhythmia_raw_clean.csv TC/dataset/
  !cd TC && python train_tc_kaggle.py
"""

import os
import sys
import json
import copy
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Fix Windows encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ================================================================
# IMPORTS TU REPO TC GOC
# ================================================================
# Them path neu can
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

# ================================================================
# CONFIG
# ================================================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Duong dan data — tu dong tim file
DATA_PATHS = [
    "dataset/Arrhythmia_raw_clean.csv",                       # Kaggle: sau khi copy
    "../IAI-Report/code/data/Arrhythmia_raw_clean.csv",       # Kaggle: relative
    "data/Arrhythmia_raw_clean.csv",                          # Chay tu thu muc code/
]

# Thu muc output
OUTPUT_DIR = "results"
MODEL_DIR = "models"

# TC Hyperparameters
CONFIG = {
    "q": 0.5,              # Component parameter (paper Fig. 3)
    "epochs": 50,          # Epochs moi Transformer (tang len 100 tren GPU)
    "batch_size": 128,
    "d_feature": 8,        # High-level features (n_A)
    "d_model": 128,        # Embedding dimension (n_M)
    "n_head": 8,
    "n_layers": 6,
    "d_k": 16,
    "d_v": 16,
    "lr_mul": 2.0,
    "warmup_steps": 4000,
    "temperature": 1.0,    # Temperature cho softmax scoring
    "dropout": 0.1,
    "max_chains": 20,
    "ood_noise_std": 2.0,  # Std cho OOD noise generation
}

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# ================================================================
# 1. DATASET
# ================================================================

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y.astype(int))
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    def __len__(self):
        return len(self.y)


def find_data_file():
    """Tim file data tu nhieu duong dan co the."""
    for path in DATA_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Khong tim thay file data! Da thu: {DATA_PATHS}\n"
        "Hay copy Arrhythmia_raw_clean.csv vao thu muc dataset/"
    )


def load_arrhythmia_data():
    """Load + split Arrhythmia data. Tra ve DataLoaders + metadata."""
    filepath = find_data_file()
    print(f"[DATA] Loading: {filepath}")

    # Load CSV (khong header, cot cuoi la label)
    data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
    X = data[:, :-1]
    y = data[:, -1].astype(int)

    n_samples, n_features = X.shape
    unique_labels = np.unique(y)
    n_classes = len(unique_labels)

    print(f"[DATA] Samples: {n_samples}, Features: {n_features}, Classes: {n_classes}")
    print(f"[DATA] Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Re-index labels ve 0-based
    label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
    y = np.array([label_map[l] for l in y])

    # Split train/test
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Normalize
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    # Handle NaN/Inf
    for arr in [train_X, test_X]:
        arr[np.isnan(arr)] = 0
        arr[np.isinf(arr)] = 0

    # Tao OOD data (3 kieu)
    rng = np.random.default_rng(SEED)

    # OOD 1: Gaussian noise
    ood_noise = rng.normal(0, CONFIG["ood_noise_std"], (len(test_X), n_features)).astype(np.float32)

    # OOD 2: Feature shuffle
    ood_shuffle = test_X.copy()
    for col in range(ood_shuffle.shape[1]):
        rng.shuffle(ood_shuffle[:, col])

    # OOD 3: Distribution shift
    ood_shift = test_X + rng.normal(2.0, 1.0, test_X.shape).astype(np.float32)

    OOD_X = np.vstack([ood_noise, ood_shuffle, ood_shift])
    OOD_y = np.zeros(len(OOD_X), dtype=int)

    print(f"[DATA] Train: {len(train_X)}, Test ID: {len(test_X)}, Test OOD: {len(OOD_X)}")

    # DataLoaders
    bs = CONFIG["batch_size"]
    trainloader = DataLoader(TabularDataset(train_X, train_y), batch_size=bs, shuffle=True)
    testloader = DataLoader(TabularDataset(test_X, test_y), batch_size=bs, shuffle=False)
    oodloader = DataLoader(TabularDataset(OOD_X, OOD_y), batch_size=bs, shuffle=False)

    y_dim = n_classes
    if y_dim == 2:
        y_dim = 3  # Giu nguyen logic repo goc

    metadata = {
        "dataset": "Arrhythmia_raw_clean",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "n_train": len(train_X),
        "n_test_id": len(test_X),
        "n_test_ood": len(OOD_X),
        "label_map": {str(k): int(v) for k, v in label_map.items()},
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }

    return trainloader, testloader, oodloader, n_features, y_dim, metadata


# ================================================================
# 2. FEATURE INDEX MANAGEMENT
# ================================================================

def init_feature_index(dim):
    w = [True for _ in range(dim)]
    s = [False for _ in range(dim)]
    return [[w], [s]]

def update_feature_index(index_list, c, weights, q):
    weights = weights.cpu().detach().numpy()
    num = int(len(weights) * (1 - q))
    if num == 0: num = 1
    order = np.argsort(weights)
    w_new = copy.deepcopy(index_list[0][c - 1])
    s_new = copy.deepcopy(index_list[1][0])
    for i in range(num):
        idx = order[i]
        cnt = -1
        for j in range(len(w_new)):
            cnt += int(index_list[0][c - 1][j])
            if cnt == idx:
                w_new[j] = False
                s_new[j] = True
                break
    index_list[0].append(w_new)
    index_list[1].append(s_new)


# ================================================================
# 3. TRAINING & SCORING
# ================================================================

def train_one(trainloader, w_idx, s_idx, net, optimizer, criterion, epochs):
    net.train()
    for epoch in range(epochs):
        loss_sum, correct, total = 0, 0, 0
        for inputs, targets in trainloader:
            data, label = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            w_data = data[:, w_idx]
            s_data = data[:, s_idx]
            out = net(w_data, s_data)
            loss = criterion(out, label)
            loss_sum += loss.item()
            _, pred = torch.max(out, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()
            loss.backward()
            optimizer.step_and_update_lr()
        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}/{epochs}: "
                  f"Loss={loss_sum/(total//CONFIG['batch_size']+1):.4f}, "
                  f"Acc={100.*correct/total:.1f}%")


def get_scores(dataloader, w_idx, s_idx, net, temp):
    """Tinh confidence scores (max softmax)."""
    net.eval()
    scores = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            data = inputs.to(DEVICE)
            out = net(data[:, w_idx], data[:, s_idx])
            probs = F.softmax(out / temp, dim=1)
            max_conf, _ = torch.max(probs, dim=1)
            scores.extend(max_conf.cpu().numpy().tolist())
    return np.array(scores)


def get_full_predictions(dataloader, w_idx, s_idx, net, temp):
    """Tra ve du doan day du: predicted class + all class probabilities."""
    net.eval()
    all_preds = []
    all_probs = []
    all_max_conf = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            data = inputs.to(DEVICE)
            out = net(data[:, w_idx], data[:, s_idx])
            probs = F.softmax(out / temp, dim=1)
            max_conf, pred = torch.max(probs, dim=1)
            all_preds.extend(pred.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_max_conf.extend(max_conf.cpu().numpy().tolist())
    return all_preds, all_probs, all_max_conf


def get_ensemble_scores(dataloader, index_list, nets, n, temp):
    """Tinh ensemble scores."""
    scores = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            data = inputs.to(DEVICE)
            ens = torch.zeros(data.size(0), device=DEVICE)
            for i in range(n):
                nets[i].eval()
                out = nets[i](data[:, index_list[0][i]], data[:, index_list[1][i]])
                probs = F.softmax(out / temp, dim=1)
                max_p, _ = torch.max(probs, dim=1)
                ens += max_p
            ens /= n
            scores.extend(ens.cpu().numpy().tolist())
    return np.array(scores)


def get_ensemble_predictions(dataloader, index_list, nets, n, temp, n_classes):
    """Tra ve ensemble predictions day du."""
    all_preds = []
    all_probs = []
    all_max_conf = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            data = inputs.to(DEVICE)
            ens_probs = torch.zeros(data.size(0), n_classes, device=DEVICE)
            for i in range(n):
                nets[i].eval()
                out = nets[i](data[:, index_list[0][i]], data[:, index_list[1][i]])
                ens_probs += F.softmax(out / temp, dim=1)
            ens_probs /= n
            max_conf, pred = torch.max(ens_probs, dim=1)
            all_preds.extend(pred.cpu().numpy().tolist())
            all_probs.extend(ens_probs.cpu().numpy().tolist())
            all_max_conf.extend(max_conf.cpu().numpy().tolist())
    return all_preds, all_probs, all_max_conf


def test_accuracy(dataloader, w_idx, s_idx, net):
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            data, label = inputs.to(DEVICE), targets.to(DEVICE)
            out = net(data[:, w_idx], data[:, s_idx])
            _, pred = torch.max(out, 1)
            total += label.size(0)
            correct += pred.eq(label).sum().item()
    return 100. * correct / total


# ================================================================
# 4. AUROC (from evaluation.py, fixed np.float)
# ================================================================

def auroc_score(id_scores, ood_scores, precision=100000):
    end = max(np.max(id_scores), np.max(ood_scores))
    start = min(np.min(id_scores), np.min(ood_scores))
    gap = (end - start) / precision
    auroc_val = 0.0
    fpr_temp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(id_scores >= delta) / float(len(id_scores))
        fpr = np.sum(ood_scores > delta) / float(len(ood_scores))
        auroc_val += (-fpr + fpr_temp) * tpr
        fpr_temp = fpr
    auroc_val += fpr * tpr
    return auroc_val * 100


def detection_error(id_scores, ood_scores, precision=100000):
    end = max(np.max(id_scores), np.max(ood_scores))
    start = min(np.min(id_scores), np.min(ood_scores))
    gap = (end - start) / precision
    err = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(id_scores < delta) / float(len(id_scores))
        err2 = np.sum(ood_scores > delta) / float(len(ood_scores))
        err = min(err, (tpr + err2) / 2.0)
    return err * 100


# ================================================================
# 5. CLINICAL LABEL (Nhan lam sang cho du doan)
# ================================================================

def clinical_label(confidence):
    """Chuyen confidence score thanh nhan lam sang."""
    if confidence >= 0.9:
        return "Rat chac chan (Very High Confidence)"
    elif confidence >= 0.7:
        return "Chac chan (High Confidence)"
    elif confidence >= 0.5:
        return "Khong chac chan - Nen hoi bac si (Uncertain - Consult Doctor)"
    else:
        return "Canh bao OOD - Du lieu bat thuong! (OOD Warning - Anomaly!)"


# ================================================================
# 6. MAIN PIPELINE
# ================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("TRANSFORMER CHAIN (TC) — TRAIN & EXPORT")
    print(f"Device: {DEVICE}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ---- LOAD DATA ----
    trainloader, testloader, oodloader, n_features, y_dim, metadata = \
        load_arrhythmia_data()

    # ---- TRAIN TC CHAIN ----
    print(f"\n{'='*60}")
    print(f"TRAINING TC CHAIN (q={CONFIG['q']}, epochs={CONFIG['epochs']})")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss()
    index_list = init_feature_index(n_features)
    nets = []
    chain_results = []
    c = 0
    temp = CONFIG["temperature"]

    while True:
        n_w = sum(index_list[0][c])
        n_s = sum(index_list[1][c])
        print(f"\n  --- Chain {c+1}: Weak={n_w}, Strong={n_s} ---")

        net = Transformer(
            src_num_inputs=n_w, trg_num_inputs=n_s,
            num_feature=CONFIG["d_feature"], num_outputs=y_dim,
            src_pad_idx=0, trg_pad_idx=0,
            d_k=CONFIG["d_k"], d_v=CONFIG["d_v"],
            d_model=CONFIG["d_model"], d_inner=0,
            n_layers=CONFIG["n_layers"], n_head=CONFIG["n_head"],
            dropout=CONFIG["dropout"], scale_emb_or_prj='prj'
        ).to(DEVICE)

        opt = ScheduledOptim(
            optim.Adam(net.parameters(), betas=(0.9, 0.98), eps=1e-09),
            CONFIG["lr_mul"], CONFIG["d_model"], CONFIG["warmup_steps"]
        )

        train_one(trainloader, index_list[0][c], index_list[1][c],
                  net, opt, criterion, CONFIG["epochs"])

        # Eval single
        acc = test_accuracy(testloader, index_list[0][c], index_list[1][c], net)
        id_sc = get_scores(testloader, index_list[0][c], index_list[1][c], net, temp)
        ood_sc = get_scores(oodloader, index_list[0][c], index_list[1][c], net, temp)

        if np.min([np.min(id_sc), np.min(ood_sc)]) == np.max([np.max(id_sc), np.max(ood_sc)]):
            print("      Score collapse, stopping.")
            break

        aur = auroc_score(id_sc, ood_sc)
        det = detection_error(id_sc, ood_sc)
        gap = float(np.mean(id_sc) - np.mean(ood_sc))

        print(f"      Acc: {acc:.1f}%, AUROC: {aur:.1f}%, DetErr: {det:.1f}%, Gap: {gap:.4f}")

        chain_results.append({
            "chain": c + 1, "weak": n_w, "strong": n_s,
            "accuracy": round(acc, 2), "auroc": round(aur, 2),
            "detection_error": round(det, 2), "confidence_gap": round(gap, 4)
        })

        nets.append(net)
        c += 1

        # Save model moi chain
        torch.save(net.state_dict(), f"{MODEL_DIR}/tc_chain_{c}.pt")
        print(f"      >> Saved: {MODEL_DIR}/tc_chain_{c}.pt")

        if n_w >= 2 and c < CONFIG["max_chains"]:
            weights = torch.norm(net.get_weights(), dim=1)
            update_feature_index(index_list, c, weights, CONFIG["q"])
        else:
            break

    n_chains = c

    # ---- ENSEMBLE ----
    print(f"\n{'='*60}")
    print(f"ENSEMBLE ({n_chains} chains)")
    print(f"{'='*60}")

    ens_id = get_ensemble_scores(testloader, index_list, nets, n_chains, temp)
    ens_ood = get_ensemble_scores(oodloader, index_list, nets, n_chains, temp)
    ens_auroc = auroc_score(ens_id, ens_ood)
    ens_det = detection_error(ens_id, ens_ood)

    print(f"  AUROC: {ens_auroc:.2f}%")
    print(f"  Detection Error: {ens_det:.2f}%")
    print(f"  Confidence — ID: {np.mean(ens_id):.4f}, OOD: {np.mean(ens_ood):.4f}")

    # ================================================================
    # OUTPUT 1: SAVE MODEL (.pt) — toan bo chain
    # ================================================================
    model_save = {
        "n_chains": n_chains,
        "config": CONFIG,
        "index_list": index_list,
        "metadata": metadata,
        "chain_results": chain_results,
    }
    for i in range(n_chains):
        model_save[f"chain_{i}_state_dict"] = nets[i].state_dict()

    model_path = f"{MODEL_DIR}/tc_ensemble_full.pt"
    torch.save(model_save, model_path)
    print(f"\n[SAVE] Model: {model_path}")

    # ================================================================
    # OUTPUT 2: SCORE CSV — cho compare.py cua Thai (A)
    # ================================================================
    # Format: 1 cot "score", moi dong la confidence score

    def save_score_csv(scores, filename):
        path = f"{OUTPUT_DIR}/{filename}"
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["score"])
            for s in scores:
                w.writerow([f"{s:.6f}"])
        print(f"[SAVE] Score CSV: {path}")

    save_score_csv(ens_id, "tc_id.csv")
    save_score_csv(ens_ood, "tc_ood.csv")

    # ================================================================
    # OUTPUT 3: JSON PREDICTIONS — chi tiet du doan + nhan lam sang
    # ================================================================
    n_classes_actual = metadata["n_classes"]
    preds_id, probs_id, confs_id = get_ensemble_predictions(
        testloader, index_list, nets, n_chains, temp, y_dim
    )
    preds_ood, probs_ood, confs_ood = get_ensemble_predictions(
        oodloader, index_list, nets, n_chains, temp, y_dim
    )

    # JSON cho ID test
    id_predictions = []
    for i, (pred, probs, conf) in enumerate(zip(preds_id, probs_id, confs_id)):
        entry = {
            "sample_id": i,
            "type": "ID",
            "predicted_class": int(pred),
            "confidence": round(float(conf) * 100, 2),  # %
            "confidence_raw": round(float(conf), 6),
            "clinical_label": clinical_label(float(conf)),
            "class_probabilities": {
                f"class_{j}": round(float(p) * 100, 2)
                for j, p in enumerate(probs)
            }
        }
        id_predictions.append(entry)

    # JSON cho OOD test
    ood_predictions = []
    for i, (pred, probs, conf) in enumerate(zip(preds_ood, probs_ood, confs_ood)):
        entry = {
            "sample_id": i,
            "type": "OOD",
            "predicted_class": int(pred),
            "confidence": round(float(conf) * 100, 2),
            "confidence_raw": round(float(conf), 6),
            "clinical_label": clinical_label(float(conf)),
            "class_probabilities": {
                f"class_{j}": round(float(p) * 100, 2)
                for j, p in enumerate(probs)
            }
        }
        ood_predictions.append(entry)

    predictions_json = {
        "model": "Transformer Chain (TC)",
        "dataset": metadata["dataset"],
        "n_chains": n_chains,
        "config": CONFIG,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "ensemble_auroc": round(ens_auroc, 2),
            "ensemble_detection_error": round(ens_det, 2),
            "avg_confidence_id": round(float(np.mean(ens_id)) * 100, 2),
            "avg_confidence_ood": round(float(np.mean(ens_ood)) * 100, 2),
            "confidence_gap": round(float(np.mean(ens_id) - np.mean(ens_ood)) * 100, 2),
        },
        "clinical_thresholds": {
            ">=90%": "Rat chac chan → Tin tuong ket qua",
            "70-89%": "Chac chan → Tin tuong nhung nen theo doi",
            "50-69%": "Khong chac chan → BAO BAC SI kiem tra lai",
            "<50%": "CANH BAO OOD → Du lieu bat thuong, KHONG tin tuong!"
        },
        "per_chain_results": chain_results,
        "id_predictions": id_predictions[:50],     # Giu 50 mau dau de file khong qua lon
        "ood_predictions": ood_predictions[:50],
    }

    json_path = f"{OUTPUT_DIR}/tc_predictions.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(predictions_json, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] Predictions JSON: {json_path}")

    # ================================================================
    # OUTPUT 4: CSV KET QUA CHI TIET
    # ================================================================
    results_csv_path = f"{OUTPUT_DIR}/tc_results_detail.csv"
    with open(results_csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        w.writerow(["Dataset", metadata["dataset"]])
        w.writerow(["N_features", n_features])
        w.writerow(["N_classes", metadata["n_classes"]])
        w.writerow(["N_chains", n_chains])
        w.writerow(["q", CONFIG["q"]])
        w.writerow(["Epochs", CONFIG["epochs"]])
        w.writerow(["Ensemble AUROC (%)", f"{ens_auroc:.2f}"])
        w.writerow(["Ensemble Detection Error (%)", f"{ens_det:.2f}"])
        w.writerow(["Avg Confidence ID (%)", f"{np.mean(ens_id)*100:.2f}"])
        w.writerow(["Avg Confidence OOD (%)", f"{np.mean(ens_ood)*100:.2f}"])
        w.writerow(["Confidence Gap (%)", f"{(np.mean(ens_id)-np.mean(ens_ood))*100:.2f}"])
        w.writerow(["---", "---"])
        for cr in chain_results:
            w.writerow([f"Chain {cr['chain']}", f"Acc={cr['accuracy']}%, AUROC={cr['auroc']}%"])
    print(f"[SAVE] Results CSV: {results_csv_path}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*60}")
    print("DONE! Output files:")
    print(f"{'='*60}")
    print(f"  [MODEL]  {MODEL_DIR}/tc_ensemble_full.pt — Toan bo {n_chains} chains")
    for i in range(n_chains):
        print(f"           {MODEL_DIR}/tc_chain_{i+1}.pt — Chain {i+1} rieng le")
    print(f"  [SCORE]  {OUTPUT_DIR}/tc_id.csv — Score ID (cho compare.py)")
    print(f"           {OUTPUT_DIR}/tc_ood.csv — Score OOD (cho compare.py)")
    print(f"  [JSON]   {OUTPUT_DIR}/tc_predictions.json — Du doan chi tiet + nhan lam sang")
    print(f"  [CSV]    {OUTPUT_DIR}/tc_results_detail.csv — Bang ket qua")
    print(f"{'='*60}")

    # Print vi du predictions
    print(f"\nVi du 5 mau ID dau tien:")
    print(f"{'Sample':>8} {'Conf%':>8} {'Class':>6} {'Label':>50}")
    for p in id_predictions[:5]:
        print(f"{p['sample_id']:>8} {p['confidence']:>7.1f}% {p['predicted_class']:>6} {p['clinical_label']:>50}")

    print(f"\nVi du 5 mau OOD dau tien:")
    for p in ood_predictions[:5]:
        print(f"{p['sample_id']:>8} {p['confidence']:>7.1f}% {p['predicted_class']:>6} {p['clinical_label']:>50}")


if __name__ == '__main__':
    main()
