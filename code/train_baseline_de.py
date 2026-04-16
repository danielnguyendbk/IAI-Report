import os
import sys
import json
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

if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass


# ================================================================
# CONFIGURATION
# ================================================================
SEED = 42
DEVICE = torch.device("cpu") # Ep chay CPU de tranh treo may tren Windows

CONFIG = {
    "epochs": 40,
    "batch_size": 32,
    "d_model": 32,   # Embedded dimension tung dac trung
    "n_head": 4,
    "n_layers": 2,
    "dropout": 0.1,
    "lr": 0.001,
    "temperature": 1.0,
    "n_ensemble": 5, # So luong model Deep Ensemble
    "ood_noise_std": 2.0
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATHS = [
    os.path.join(BASE_DIR, "Arrhythmia_raw_clean.csv"),
    os.path.join(BASE_DIR, "data", "Arrhythmia_raw_clean.csv"),
    "Arrhythmia_raw_clean.csv"
]
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ================================================================
# DATASET & LOADER
# ================================================================
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y.astype(int))
    def __getitem__(self, idx): return self.X[idx], self.y[idx]
    def __len__(self): return len(self.y)

def load_data():
    filepath = next((p for p in DATA_PATHS if os.path.exists(p)), None)
    if not filepath: raise FileNotFoundError(f"Khong tim thay CSV! Cac duong dan da thu: {DATA_PATHS}")
    
    print(f"[*] Tai du lieu tu file .../{os.path.basename(filepath)}")
    data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
    X = data[:, :-1]
    y = data[:, -1].astype(int)

    n_samples, n_features = X.shape
    unique_labels = np.unique(y)
    n_classes = len(unique_labels)
    
    # Reindex nhan
    label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
    y = np.array([label_map[l] for l in y])

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    for arr in [train_X, test_X]:
        arr[np.isnan(arr)] = 0
        arr[np.isinf(arr)] = 0

    rng = np.random.default_rng(SEED)
    ood_noise = rng.normal(0, CONFIG["ood_noise_std"], (len(test_X), n_features)).astype(np.float32)
    OOD_y = np.zeros(len(ood_noise), dtype=int)

    trainloader = DataLoader(TabularDataset(train_X, train_y), batch_size=CONFIG["batch_size"], shuffle=True)
    testloader = DataLoader(TabularDataset(test_X, test_y), batch_size=CONFIG["batch_size"], shuffle=False)
    oodloader = DataLoader(TabularDataset(ood_noise, OOD_y), batch_size=CONFIG["batch_size"], shuffle=False)

    y_dim = n_classes if n_classes > 2 else 3 # Tuong thich voi repo goc
    return trainloader, testloader, oodloader, n_features, y_dim

# ================================================================
# MODEL: SINGLE TRANSFORMER FOR TABULAR
# ================================================================
class SingleTransformer(nn.Module):
    def __init__(self, n_features, n_classes, d_model=32, n_head=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.feature_embed = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: (batch, n_features) -> (batch, n_features, 1)
        x = x.unsqueeze(-1)
        x = self.feature_embed(x) # (batch, n_features, d_model)
        
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (batch, n_f+1, d_model)
        x = self.transformer(x)
        
        # Lay cls_token de phan loai
        cls_out = x[:, 0, :] 
        return self.fc_out(cls_out)


def train_model(net, trainloader, epochs, seed_val):
    torch.manual_seed(seed_val)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=CONFIG["lr"])
    
    net.train()
    for ep in range(epochs):
        loss_val = 0.0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            out = net(inputs)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
        if (ep + 1) % 10 == 0:
            print(f"      -> Epoch {ep+1}/{epochs} | Loss: {loss_val:.4f}")
    return net

def get_max_conf_scores(net, dataloader):
    net.eval()
    scores = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(DEVICE)
            out = net(inputs)
            probs = F.softmax(out / CONFIG["temperature"], dim=1)
            max_conf, _ = torch.max(probs, dim=1)
            scores.extend(max_conf.cpu().numpy().tolist())
    return np.array(scores)

# ================================================================
# MAIN PROGRAM
# ================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("=" * 60)
    print(" TRAIN BASELINE & DEEP ENSEMBLE (Chay cuc ky nhanh tren Local)")
    print("=" * 60)
    
    trainloader, testloader, oodloader, n_features, y_dim = load_data()
    
    print("\n[1] TRAIN BASELINE (Single Transformer)...")
    baseline_net = SingleTransformer(n_features, y_dim, **{k:v for k,v in CONFIG.items() if k in ['d_model','n_head','n_layers','dropout']}).to(DEVICE)
    baseline_net = train_model(baseline_net, trainloader, CONFIG["epochs"], SEED)
    
    base_id_scores = get_max_conf_scores(baseline_net, testloader)
    base_ood_scores = get_max_conf_scores(baseline_net, oodloader)
    
    torch.save(baseline_net.state_dict(), f"{MODEL_DIR}/baseline.pt")
    print(f" >> Hoan thanh Baseline. Da luu tai {MODEL_DIR}/baseline.pt")
    
    print("\n[2] TRAIN DEEP ENSEMBLE (5 Transformers parallel)...")
    ensemble_nets = []
    for i in range(CONFIG["n_ensemble"]):
        print(f"    - Dang train Transformer thanh phan #{i+1} / {CONFIG['n_ensemble']}...")
        net = SingleTransformer(n_features, y_dim, **{k:v for k,v in CONFIG.items() if k in ['d_model','n_head','n_layers','dropout']}).to(DEVICE)
        # Train voi seed khac nhau de tao su Da dang (Diversity) trong Ensemble
        net = train_model(net, trainloader, CONFIG["epochs"], SEED + i + 1)
        ensemble_nets.append(net)
        
    print(" >> Luu thong tin toan bo Ensemble models...")
    ensemble_state = {"n_ensemble": CONFIG["n_ensemble"], "config": CONFIG}
    for i, net in enumerate(ensemble_nets):
        ensemble_state[f"model_{i}"] = net.state_dict()
    torch.save(ensemble_state, f"{MODEL_DIR}/de_ensemble_full.pt")
    
    def get_ensemble_scores(nets, dataloader):
        scores = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(DEVICE)
                ens_probs = torch.zeros(inputs.size(0), y_dim, device=DEVICE)
                for net in nets:
                    net.eval()
                    out = net(inputs)
                    ens_probs += F.softmax(out / CONFIG["temperature"], dim=1)
                ens_probs /= len(nets)
                max_conf, _ = torch.max(ens_probs, dim=1)
                scores.extend(max_conf.cpu().numpy().tolist())
        return np.array(scores)

    de_id_scores = get_ensemble_scores(ensemble_nets, testloader)
    de_ood_scores = get_ensemble_scores(ensemble_nets, oodloader)
    
    print("\n[3] XUAT FILE SCORE CSV THEO CHUAN (Cua Hiep)...")
    def save_csv(scores, filename):
        with open(f"{OUTPUT_DIR}/{filename}", 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["score"])
            for s in scores: w.writerow([f"{s:.6f}"])
            
    save_csv(base_id_scores, "baseline_id.csv")
    save_csv(base_ood_scores, "baseline_ood.csv")
    save_csv(de_id_scores, "de_id.csv")
    save_csv(de_ood_scores, "de_ood.csv")
    
    print("\n[HOAN THANH] Xac nhan nhat ky hoat dong:")
    print(" [+] Luu models: models/baseline.pt, models/de_ensemble_full.pt")
    print(" [+] Luu ket qua: baseline_id.csv, baseline_ood.csv, de_id.csv, de_ood.csv")

if __name__ == "__main__":
    main()
