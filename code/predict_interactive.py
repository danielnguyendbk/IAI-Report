"""
predict_interactive.py — Du doan benh nhan TUONG TAC
====================================================
Chay: python predict_interactive.py
Menu:
  1. Nhap tay (paste 279 so cach nhau bang dau phay)
  2. Tai file CSV
  3. Chay mau demo
  4. Thoat
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F

# Fix Windows encoding
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

# Import Transformer
for p in ['d:/research/TC', '.', '..', '../TC']:
    if os.path.exists(os.path.join(p, 'transformer', 'Models.py')):
        sys.path.insert(0, os.path.abspath(p))
        break
from transformer.Models import Transformer

# ================================================================
# CONFIG
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "tc_ensemble_full.pt")
for mp in [MODEL_PATH, "models/tc_ensemble_full.pt",
           "tc_output/models/tc_ensemble_full.pt",
           "../tc_output/models/tc_ensemble_full.pt"]:
    if os.path.exists(mp):
        MODEL_PATH = mp
        break

DEVICE = torch.device("cpu")
THRESHOLD_HIGH = 34.7
THRESHOLD_MID = 20.8
THRESHOLD_LOW = 11.5

ARRHYTHMIA_NAMES = {
    0: "Binh thuong (Normal)",
    1: "Thieu mau cuc bo (Ischemic changes)",
    2: "Nhip tim som (Old Anterior MI)",
    3: "Nhip tim som (Old Inferior MI)",
    4: "Xoang bat thuong (Sinus tachycardia)",
    5: "Xoang bat thuong (Sinus bradycardia)",
    6: "Co that som (Ventricular premature)",
    7: "Co that som tren (Supraventricular premature)",
    8: "Block nhanh trai (Left bundle branch block)",
    9: "Block nhanh phai (Right bundle branch block)",
    10: "Block AV do 1 (1st degree AV block)",
    11: "Block AV do 2 (2nd degree AV block)",
    12: "Block AV do 3 (3rd degree AV block)",
}


# ================================================================
# LOAD MODEL
# ================================================================
def load_tc_model(model_path):
    print(f"\n  Dang tai model: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    n_chains = checkpoint["n_chains"]
    config = checkpoint["config"]
    index_list = checkpoint["index_list"]
    metadata = checkpoint["metadata"]

    y_dim = metadata["n_classes"]
    if y_dim == 2:
        y_dim = 3

    nets = []
    for i in range(n_chains):
        n_w = sum(index_list[0][i])
        n_s = sum(index_list[1][i])
        net = Transformer(
            src_num_inputs=n_w, trg_num_inputs=n_s,
            num_feature=config["d_feature"], num_outputs=y_dim,
            src_pad_idx=0, trg_pad_idx=0,
            d_k=config["d_k"], d_v=config["d_v"],
            d_model=config["d_model"], d_inner=0,
            n_layers=config["n_layers"], n_head=config["n_head"],
            dropout=config["dropout"], scale_emb_or_prj='prj'
        ).to(DEVICE)
        net.load_state_dict(checkpoint[f"chain_{i}_state_dict"])
        net.eval()
        nets.append(net)

    print(f"  Tai thanh cong! {n_chains} chains, {metadata['n_features']} features, {metadata['n_classes']} classes")
    return nets, index_list, n_chains, config, metadata


# ================================================================
# PREDICT
# ================================================================
def predict_single(patient_data, nets, index_list, n_chains, config, n_classes):
    data = torch.FloatTensor(patient_data).unsqueeze(0).to(DEVICE)
    temp = config["temperature"]
    y_dim = n_classes if n_classes > 2 else 3

    ens_probs = torch.zeros(1, y_dim, device=DEVICE)
    with torch.no_grad():
        for i in range(n_chains):
            nets[i].eval()
            w_idx = index_list[0][i]
            s_idx = index_list[1][i]
            out = nets[i](data[:, w_idx], data[:, s_idx])
            ens_probs += F.softmax(out / temp, dim=1)
    ens_probs /= n_chains

    max_conf, pred_class = torch.max(ens_probs, dim=1)
    confidence = float(max_conf.item()) * 100
    predicted = int(pred_class.item())
    all_probs = ens_probs.squeeze().cpu().numpy() * 100
    return predicted, confidence, all_probs


def clinical_label(confidence):
    if confidence >= THRESHOLD_HIGH:
        return "[OK] BINH THUONG -- AI tu tin, du lieu hop le"
    elif confidence >= THRESHOLD_MID:
        return "[??] CAN THEO DOI -- Nen kiem tra them"
    elif confidence >= THRESHOLD_LOW:
        return "[!!] KHONG CHAC CHAN -- BAO BAC SI kiem tra lai!"
    else:
        return "[XX] CANH BAO OOD -- Du lieu BAT THUONG, KHONG tin tuong!"


def print_result(patient_name, predicted, confidence, all_probs):
    print(f"\n  {'='*55}")
    print(f"  KET QUA: {patient_name}")
    print(f"  {'='*55}")
    cl = clinical_label(confidence)
    print(f"\n  >>> {cl}")
    print(f"\n  Du doan:    {ARRHYTHMIA_NAMES.get(predicted, f'Class {predicted}')}")
    print(f"  Confidence: {confidence:.1f}%")

    top3 = np.argsort(all_probs)[::-1][:3]
    print(f"\n  Top 3 kha nang:")
    for rank, cls in enumerate(top3, 1):
        name = ARRHYTHMIA_NAMES.get(cls, f"Class {cls}")
        print(f"    {rank}. {name}: {all_probs[cls]:.1f}%")

    bar_len = 30
    filled = int(confidence / 100 * bar_len)
    bar = "#" * filled + "." * (bar_len - filled)
    print(f"\n  [{bar}] {confidence:.1f}%")
    print(f"  <11.5%=OOD | 11.5-20.8%=Hoi BS | 20.8-34.7%=Theo doi | >=34.7%=OK")


# ================================================================
# GET SCALER (tu dataset goc)
# ================================================================
def get_scaler():
    """Load dataset goc de tinh mean/std cho StandardScaler."""
    from sklearn.preprocessing import StandardScaler
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_paths = [
        os.path.join(BASE_DIR, "Arrhythmia_raw_clean.csv"),
        os.path.join(BASE_DIR, "data", "Arrhythmia_raw_clean.csv"),
        "dataset/Arrhythmia_raw_clean.csv",
    ]
    for p in data_paths:
        if os.path.exists(p):
            raw = np.loadtxt(p, delimiter=',', dtype=np.float32)
            X = raw[:, :-1]
            scaler = StandardScaler()
            scaler.fit(X)
            return scaler, X
    return None, None


# ================================================================
# MENU FUNCTIONS
# ================================================================
def option_manual_input(nets, index_list, n_chains, config, metadata, scaler):
    """Nhap tay du lieu benh nhan."""
    n_features = metadata["n_features"]
    n_classes = metadata["n_classes"]

    print(f"\n  {'='*55}")
    print(f"  NHAP TAY DU LIEU BENH NHAN")
    print(f"  {'='*55}")
    print(f"  Can nhap {n_features} gia tri, cach nhau bang dau phay.")
    print(f"  Vi du dong dau cua dataset:")
    print(f"  75,0,190,80,91,193,371,174,121,-16,13,64,-2,-50.5,...")
    print(f"")
    print(f"  Hoac nhap 'demo1' de dung mau binh thuong")
    print(f"  Hoac nhap 'demo2' de dung mau benh")
    print(f"  Hoac nhap 'random' de tao du lieu ngau nhien (OOD)")
    print(f"  Nhap 'back' de quay lai menu")

    while True:
        print(f"\n  ---")
        user_input = input("  Nhap du lieu (hoac demo1/demo2/random/back): ").strip()

        if user_input.lower() == 'back':
            return

        patient_data = None
        patient_name = "Benh nhan nhap tay"
        need_scale = True

        if user_input.lower() == 'demo1':
            # Mau binh thuong tu dataset
            if scaler is not None:
                BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                data_pth = os.path.join(BASE_DIR, "Arrhythmia_raw_clean.csv")
                b_pth = os.path.join(BASE_DIR, "data", "Arrhythmia_raw_clean.csv")
                dp = data_pth if os.path.exists(data_pth) else b_pth if os.path.exists(b_pth) else "Arrhythmia_raw_clean.csv"
                try:
                    raw = np.loadtxt(dp, delimiter=',', dtype=np.float32)
                    normal_idx = np.where(raw[:, -1] == 1)[0]
                    patient_data = raw[normal_idx[0], :-1]
                    patient_name = "MAU DEMO - Benh nhan binh thuong"
                except:
                    print("  [LOI] Khong tim thay dataset de lay mau!")
                    continue
            else:
                print("  [LOI] Khong tim thay dataset de lay mau!")
                continue

        elif user_input.lower() == 'demo2':
            # Mau co benh tu dataset
            if scaler is not None:
                BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                data_pth = os.path.join(BASE_DIR, "Arrhythmia_raw_clean.csv")
                b_pth = os.path.join(BASE_DIR, "data", "Arrhythmia_raw_clean.csv")
                dp = data_pth if os.path.exists(data_pth) else b_pth if os.path.exists(b_pth) else "Arrhythmia_raw_clean.csv"
                try:
                    raw = np.loadtxt(dp, delimiter=',', dtype=np.float32)
                    sick_idx = np.where(raw[:, -1] == 10)[0]
                    if len(sick_idx) > 0:
                        patient_data = raw[sick_idx[0], :-1]
                        patient_name = "MAU DEMO - Benh nhan co benh (Block nhanh phai)"
                    else:
                        print("  [LOI] Khong tim thay mau benh!")
                        continue
                except:
                    print("  [LOI] Khong tim thay mau benh!")
                    continue
            else:
                print("  [LOI] Khong tim thay dataset de lay mau!")
                continue

        elif user_input.lower() == 'random':
            # Du lieu ngau nhien (OOD)
            rng = np.random.default_rng()
            patient_data = rng.normal(0, 2.0, n_features).astype(np.float32)
            patient_name = "DU LIEU NGAU NHIEN (OOD test)"
            need_scale = False  # Da la random, khong can scale

        else:
            # Parse input cua nguoi dung
            try:
                values = user_input.replace('\n', ',').replace('\t', ',')
                parts = [x.strip() for x in values.split(',') if x.strip()]
                patient_data = np.array([float(x) for x in parts], dtype=np.float32)

                if len(patient_data) == n_features + 1:
                    # Co cot label o cuoi → bo di
                    patient_data = patient_data[:-1]
                    print(f"  (Bo cot label cuoi, con {n_features} features)")

                if len(patient_data) != n_features:
                    print(f"  [LOI] Can {n_features} gia tri, ban nhap {len(patient_data)}!")
                    print(f"  Hay nhap lai hoac dung 'demo1'/'demo2' de test.")
                    continue

                patient_name = "Benh nhan nhap tay"

            except ValueError as e:
                print(f"  [LOI] Khong doc duoc so: {e}")
                print(f"  Hay nhap cac so cach nhau bang dau phay.")
                continue

        # Scale du lieu
        if need_scale and scaler is not None:
            patient_data = scaler.transform(patient_data.reshape(1, -1)).flatten()
        elif need_scale:
            print("  [CANH BAO] Khong co scaler, du lieu chua chuan hoa!")

        # Predict
        pred, conf, probs = predict_single(
            patient_data, nets, index_list, n_chains, config, n_classes)
        print_result(patient_name, pred, conf, probs)


def option_csv_file(nets, index_list, n_chains, config, metadata, scaler):
    """Doc file CSV."""
    n_features = metadata["n_features"]
    n_classes = metadata["n_classes"]

    print(f"\n  {'='*55}")
    print(f"  TAI FILE CSV")
    print(f"  {'='*55}")
    print(f"  File CSV can co {n_features} cot (hoac {n_features+1} cot voi label)")
    print(f"  Moi dong la 1 benh nhan")

    csv_path = input("\n  Nhap duong dan file CSV (hoac 'back'): ").strip()
    if csv_path.lower() == 'back':
        return

    # Xu ly duong dan co dau " hoac '
    csv_path = csv_path.strip('"').strip("'")

    if not os.path.exists(csv_path):
        print(f"  [LOI] Khong tim thay file: {csv_path}")
        return

    try:
        data = np.loadtxt(csv_path, delimiter=',', dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Bo cot label neu co
        if data.shape[1] == n_features + 1:
            print(f"  Phat hien {data.shape[1]} cot, bo cot label cuoi")
            data = data[:, :-1]

        if data.shape[1] != n_features:
            print(f"  [LOI] File co {data.shape[1]} cot, can {n_features}!")
            return

        print(f"  Doc thanh cong: {len(data)} benh nhan")

        # Scale
        if scaler is not None:
            data = scaler.transform(data)
        else:
            print("  [CANH BAO] Khong co scaler!")

        # Predict tung dong
        results = []
        for i in range(len(data)):
            pred, conf, probs = predict_single(
                data[i], nets, index_list, n_chains, config, n_classes)
            print_result(f"Benh nhan #{i+1}", pred, conf, probs)
            results.append({
                "patient_id": i+1,
                "predicted_class": pred,
                "disease": ARRHYTHMIA_NAMES.get(pred, f"Class {pred}"),
                "confidence": round(conf, 2),
                "clinical_label": clinical_label(conf),
            })

        # Bang tong ket
        print(f"\n  {'='*55}")
        print(f"  TONG KET: {len(results)} benh nhan")
        print(f"  {'='*55}")
        print(f"  {'#':<4} {'Conf':>6} {'Ket qua':<12} {'Du doan':<35}")
        print(f"  {'--':<4} {'----':>6} {'--------':<12} {'-'*35}")
        for r in results:
            tag = "[OK]" if r["confidence"] >= THRESHOLD_HIGH else \
                  "[??]" if r["confidence"] >= THRESHOLD_MID else \
                  "[!!]" if r["confidence"] >= THRESHOLD_LOW else "[XX]"
            print(f"  {r['patient_id']:<4} {r['confidence']:>5.1f}% {tag:<12} {r['disease'][:35]}")

        # Luu JSON
        out_path = csv_path.replace('.csv', '_predictions.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved: {out_path}")

    except Exception as e:
        print(f"  [LOI] {e}")


def option_demo(nets, index_list, n_chains, config, metadata, scaler):
    """Chay mau demo."""
    n_classes = metadata["n_classes"]

    raw = None
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(BASE_DIR, "Arrhythmia_raw_clean.csv"),
              os.path.join(BASE_DIR, "data", "Arrhythmia_raw_clean.csv"),
              "dataset/Arrhythmia_raw_clean.csv"]:
        if os.path.exists(p):
            raw = np.loadtxt(p, delimiter=',', dtype=np.float32)
            break
    if raw is None:
        print("  [LOI] Khong tim thay dataset!")
        return

    X = raw[:, :-1]
    y = raw[:, -1].astype(int)
    rng = np.random.default_rng(42)

    # Scale
    X_scaled = scaler.transform(X) if scaler else X

    # Chon mau
    samples = [
        ("Benh nhan binh thuong #1", X_scaled[np.where(y==1)[0][0]], "ID"),
        ("Benh nhan binh thuong #2", X_scaled[np.where(y==1)[0][10]], "ID"),
        ("Benh nhan co benh (Ischemic)", X_scaled[np.where(y==2)[0][0]], "ID"),
    ]
    sick10 = np.where(y==10)[0]
    if len(sick10) > 0:
        samples.append(("Benh nhan co benh (RBBB)", X_scaled[sick10[0]], "ID"))

    # OOD samples
    noise = rng.normal(0, 2.0, 279).astype(np.float32)
    samples.append(("DU LIEU NHIEU (OOD)", noise, "OOD"))

    shuffled = X_scaled[np.where(y==1)[0][0]].copy()
    rng.shuffle(shuffled)
    samples.append(("DU LIEU HOAN DOI (OOD)", shuffled, "OOD"))

    print(f"\n  {'='*55}")
    print(f"  DEMO: {len(samples)} mau")
    print(f"  {'='*55}")

    results = []
    for name, data, stype in samples:
        pred, conf, probs = predict_single(
            data, nets, index_list, n_chains, config, n_classes)
        print_result(name, pred, conf, probs)
        results.append((name, stype, conf))

    # Tong ket
    print(f"\n  {'='*55}")
    print(f"  TONG KET DEMO")
    print(f"  {'='*55}")
    for name, stype, conf in results:
        tag = "[OK]" if conf >= THRESHOLD_HIGH else \
              "[??]" if conf >= THRESHOLD_MID else \
              "[!!]" if conf >= THRESHOLD_LOW else "[XX]"
        print(f"  {tag} {conf:>5.1f}% | {stype:<4} | {name}")


# ================================================================
# MAIN MENU
# ================================================================
def main():
    print("=" * 58)
    print("  TRANSFORMER CHAIN -- HE THONG DU DOAN LOAN NHIP TIM")
    print("  (OOD Detection for Arrhythmia)")
    print("=" * 58)

    # Load model
    nets, index_list, n_chains, config, metadata = load_tc_model(MODEL_PATH)
    n_classes = metadata["n_classes"]

    # Load scaler
    scaler, _ = get_scaler()
    if scaler:
        print("  Scaler: OK (tu dataset goc)")
    else:
        print("  [CANH BAO] Khong tim thay dataset de tao scaler!")

    print(f"\n  Nguong lam sang ({n_classes} classes):")
    print(f"    >= {THRESHOLD_HIGH}%: [OK] Binh thuong")
    print(f"    >= {THRESHOLD_MID}%:  [??] Can theo doi")
    print(f"    >= {THRESHOLD_LOW}%:  [!!] Bao bac si")
    print(f"    <  {THRESHOLD_LOW}%:  [XX] CANH BAO OOD!")

    while True:
        print(f"\n  {'='*55}")
        print("  MENU:")
        print("    1. Nhap tay du lieu benh nhan")
        print("    2. Tai file CSV")
        print("    3. Chay mau demo")
        print("    4. Thoat")
        print(f"  {'='*55}")

        choice = input("  Chon (1/2/3/4): ").strip()

        if choice == '1':
            option_manual_input(nets, index_list, n_chains, config, metadata, scaler)
        elif choice == '2':
            option_csv_file(nets, index_list, n_chains, config, metadata, scaler)
        elif choice == '3':
            option_demo(nets, index_list, n_chains, config, metadata, scaler)
        elif choice == '4':
            print("\n  Tam biet!")
            break
        else:
            print("  Chon 1, 2, 3 hoac 4!")


if __name__ == "__main__":
    main()
