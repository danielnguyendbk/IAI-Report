"""
predict_baseline_de.py — Du doan benh nhan de danh gia Overconfidence cua Baseline vs DE
====================================================
Chay: python predict_baseline_de.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Cai dat chung
DEVICE = torch.device("cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATHS = [
    os.path.join(BASE_DIR, "Arrhythmia_raw_clean.csv"),
    os.path.join(BASE_DIR, "data", "Arrhythmia_raw_clean.csv"),
    "Arrhythmia_raw_clean.csv"
]

# Thu vien ten Benh Tim
ARRHYTHMIA_NAMES = {
    0: "Binh thuong (Normal)", 1: "Thieu mau cuc bo", 2: "Nhip tim som (Old Anterior MI)",
    3: "Nhip tim som (Old Inferior MI)", 4: "Xoang bat thuong (Tachycardia)", 5: "Xoang bat thuong (Bradycardia)",
    6: "Co that som (Ventricular premature)", 7: "Co that som tren (Supraventricular premature)", 
    8: "Block nhanh trai (LBBB)", 9: "Block nhanh phai (RBBB)", 10: "Block AV do 1", 
    11: "Block AV do 2", 12: "Block AV do 3"
}

# Ngưỡng lâm sàng (Chung mâm với file An)
THRESHOLD_HIGH = 34.7
THRESHOLD_MID = 20.8
THRESHOLD_LOW = 11.5

def clinical_label(confidence):
    if confidence >= THRESHOLD_HIGH: return "[OK] BINH THUONG -- Du lieu an toan do tin cay cao"
    elif confidence >= THRESHOLD_MID: return "[??] CAN THEO DOI -- Phan tan ket qua hoac bat dong mo hinh"
    elif confidence >= THRESHOLD_LOW: return "[!!] KHONG CHAC CHAN -- Muc do rui ro cao can can thiep!"
    else: return "[XX] CANH BAO OOD -- Du lieu ngoai phan phoi (Out-of-Distribution)"

# ================================================================
# MODEL ARCHITECTURE (Phai trung khop voi train_baseline_de.py)
# ================================================================
class SingleTransformer(nn.Module):
    def __init__(self, n_features, n_classes, d_model=32, n_head=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.feature_embed = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.feature_embed(x)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        return self.fc_out(x[:, 0, :])

# ================================================================
# TAI MODEL & SCALER
# ================================================================
def load_models():
    n_features, n_classes = 279, 13
    baseline = SingleTransformer(n_features, n_classes).to(DEVICE)
    baseline.load_state_dict(torch.load(f"{MODEL_DIR}/baseline.pt", map_location=DEVICE, weights_only=False))
    baseline.eval()

    ensemble_nets = []
    checkpoint = torch.load(f"{MODEL_DIR}/de_ensemble_full.pt", map_location=DEVICE, weights_only=False)
    for i in range(checkpoint["n_ensemble"]):
        net = SingleTransformer(n_features, n_classes).to(DEVICE)
        net.load_state_dict(checkpoint[f"model_{i}"])
        net.eval()
        ensemble_nets.append(net)
        
    return baseline, ensemble_nets, n_features, n_classes

def get_scaler():
    from sklearn.preprocessing import StandardScaler
    filepath = next((p for p in DATA_PATHS if os.path.exists(p)), None)
    if filepath:
        raw = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        X = raw[:, :-1]
        scaler = StandardScaler()
        scaler.fit(X)
        return scaler, X
    return None, None

# ================================================================
# PREDICTION LOGIC
# ================================================================
def predict_compare(data_row, baseline_net, ensemble_nets):
    data = torch.FloatTensor(data_row).unsqueeze(0).to(DEVICE)
    
    # 1. Baseline Predict
    with torch.no_grad():
        out_base = baseline_net(data)
        probs_b = F.softmax(out_base, dim=1)
        conf_b, pred_b = torch.max(probs_b, dim=1)
    
    # 2. Ensemble Predict
    ens_probs = torch.zeros(1, 13, device=DEVICE)
    with torch.no_grad():
        for net in ensemble_nets:
            out_e = net(data)
            ens_probs += F.softmax(out_e, dim=1)
        ens_probs /= len(ensemble_nets)
        conf_e, pred_e = torch.max(ens_probs, dim=1)
        
    return (int(pred_b), float(conf_b)*100), (int(pred_e), float(conf_e)*100)

# ================================================================
# MENU & DEMO
# ================================================================
def print_comparison(title, result_base, result_ens):
    pred_b, conf_b = result_base
    pred_e, conf_e = result_ens
    
    print(f"\n  {'='*60}")
    print(f"  KET QUA: {title}")
    print(f"  {'='*60}")
    
    # VE KET QUA BASELINE
    print(f"\n  [BASELINE - Single Transformer]")
    print(f"  >>> {clinical_label(conf_b)}")
    print(f"  Du doan:    {ARRHYTHMIA_NAMES[pred_b]}")
    bar_b = "#" * int(conf_b / 100 * 30) + "." * (30 - int(conf_b / 100 * 30))
    print(f"  [{bar_b}] {conf_b:.1f}%")
    
    # VE KET QUA ENSEMBLE
    print(f"\n  [DEEP ENSEMBLE - Hop Nhat 5 Transformers]")
    print(f"  >>> {clinical_label(conf_e)}")
    print(f"  Du doan:    {ARRHYTHMIA_NAMES[pred_e]}")
    bar_e = "#" * int(conf_e / 100 * 30) + "." * (30 - int(conf_e / 100 * 30))
    print(f"  [{bar_e}] {conf_e:.1f}%")

def option_demo(baseline_net, ensemble_nets, scaler, X_raw):
    rng = np.random.default_rng(42)
    # Mau ID (Benh that)
    X_scaled = scaler.transform(X_raw)
    sample_id = X_scaled[20] # Lay dai 1 khach binh thuong
    res_b, res_e = predict_compare(sample_id, baseline_net, ensemble_nets)
    print_comparison("MAU CHUAN ID - Nguoi Benh That", res_b, res_e)
    
    # Mau OOD - Nhieu
    noise = rng.normal(0, 2.0, 279).astype(np.float32)
    res_b, res_e = predict_compare(noise, baseline_net, ensemble_nets)
    print_comparison("MAU RÁC CỐ Ý (OOD Gaussian)", res_b, res_e)
    
    # Mau OOD - Xao tron
    shuffled = X_scaled[30].copy()
    rng.shuffle(shuffled)
    res_b, res_e = predict_compare(shuffled, baseline_net, ensemble_nets)
    print_comparison("MAU XO TRON LOI (OOD Shuffled)", res_b, res_e)
    

def main():
    print("\n==========================================================")
    print("  HE THONG DANH GIA: BASELINE VS DEEP ENSEMBLE")
    print("  (Phan tich Out-of-Distribution - Dataset Arrhythmia)")
    print("==========================================================")
    print("\n  Nguong phan tich do tu tin (13 classes):")
    print("    >= 34.7%: [OK] Binh thuong")
    print("    >= 20.8%:  [??] Can theo doi")
    print("    >= 11.5%:  [!!] Bao bac si")
    print("    <  11.5%:  [XX] CANH BAO OOD!")
    print("----------------------------------------------------------")
    
    if not os.path.exists(f"{MODEL_DIR}/baseline.pt"):
        print("[!] Khong tim thay model. Ban can chay `python train_baseline_de.py` truoc tien!")
        return

    baseline, ensemble_nets, _, _ = load_models()
    scaler, X_raw = get_scaler()
    
    while True:
        print("\nCHON CHE DO:")
        print("  1. Chay cac hinh mau Demo san (Id & OOD)")
        print("  2. Thoat")
        choice = input("Chon (1/2): ")
        if choice == '1': option_demo(baseline, ensemble_nets, scaler, X_raw)
        elif choice == '2': break
        
if __name__ == "__main__":
    main()
