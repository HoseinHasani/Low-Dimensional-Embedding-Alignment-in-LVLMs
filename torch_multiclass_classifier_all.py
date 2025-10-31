import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# ------------------ CONFIG ------------------

data_dir = "data/all layers all attention tp fp"
base_save_dir = "results_multiclass"
os.makedirs(base_save_dir, exist_ok=True)

dataset_path = "cls_data_multiclass"

n_files = 3960
n_layers, n_heads = 32, 32
min_position = 5
max_position = 150
position_margin = 2
n_top_k = 20
n_subtokens = 1
eps = 1e-10

use_entropy = True
use_gini = False

train_size = 0.75
test_size = 0.25

n_epochs = 3
weight_decay = 1e-3
dropout_rate = 0.5
normalize_features = True

fp_replication_factor = 2  # Replication factor for balancing FP class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# ------------------ UTILS ------------------

def compute_entropy(values):
    if len(values) == 0:
        return 0.0
    prob = np.array(values, dtype=float)
    prob = prob / (np.sum(prob, axis=-1, keepdims=True) + eps)
    return -np.sum(prob * np.log(prob + eps), axis=-1)

def compute_gini(values):
    if len(values) == 0:
        return 0.0
    prob = np.array(values, dtype=float)
    prob = prob / (np.sum(prob, axis=-1, keepdims=True) + eps)
    sorted_p = np.sort(prob, axis=-1)
    n = sorted_p.shape[-1]
    coef = 2 * np.arange(1, n + 1) - n - 1
    gini = np.sum(coef * sorted_p, axis=-1) / (n * np.sum(sorted_p, axis=-1) + eps)
    return np.abs(gini)

def extract_attention_values(data_dict, cls_):
    results = []
    entries = data_dict.get(cls_, {}).get("image", [])
    for e in entries:
        if not e.get("subtoken_results"):
            continue
        for sub in e["subtoken_results"][:n_subtokens]:
            topk_vals = np.array(sub["topk_values"], dtype=float)
            if topk_vals.ndim != 3:
                continue
            idx = int(sub["idx"])
            results.append((idx, topk_vals[..., :n_top_k]))
    return results

def balance_fp_samples(X, y, pos, cls, factor=5):
    fp_mask = (y == 1)  # Mask where the class is FP
    X_fp, y_fp, pos_fp, cls_fp = X[fp_mask], y[fp_mask], pos[fp_mask], cls[fp_mask]
    
    # Replicate FP samples based on the factor
    X_bal = np.concatenate([X, np.repeat(X_fp, factor, axis=0)], axis=0)
    y_bal = np.concatenate([y, np.repeat(y_fp, factor, axis=0)], axis=0)
    pos_bal = np.concatenate([pos, np.repeat(pos_fp, factor, axis=0)], axis=0)
    cls_bal = np.concatenate([cls, np.repeat(cls_fp, factor, axis=0)], axis=0)
    
    return X_bal, y_bal, pos_bal, cls_bal

# ------------------ FEATURE EXTRACTION ------------------

def extract_all_features(files, n_files, n_layers, n_heads, min_position, max_position):
    X, y, pos_list, cls_list = [], [], [], []
    for f in tqdm(files[:n_files], desc="Extracting features"):
        try:
            with open(f, "rb") as handle:
                data_dict = pickle.load(handle)
        except Exception:
            continue

        for cls_, label in [("fp", 0), ("tp", 1), ("other", 2)]:
            attention_samples = extract_attention_values(data_dict, cls_)
            for idx, topk_arr in attention_samples:
                token_pos = int(idx)
                if token_pos < min_position or token_pos > max_position:
                    continue
                features = []
                for l in range(n_layers):
                    for h in range(n_heads):
                        vals = topk_arr[l, h, :]
                        mean_attention = np.mean(vals)
                        features.append(mean_attention)
                        if use_entropy:
                            features.append(compute_entropy(vals))
                        if use_gini:
                            features.append(compute_gini(vals))
                features.append(token_pos / max_position)
                X.append(features)
                y.append(label)
                pos_list.append(token_pos)
                cls_list.append(cls_)
    if len(X) == 0:
        return None, None, None, None
    return np.array(X), np.array(y), np.array(pos_list), np.array(cls_list)


files = sorted(glob(os.path.join(data_dir, "attentions_*.pkl")))

if dataset_path and os.path.exists(f"{dataset_path}/x.npy"):
    print("Loading saved dataset...")
    X_all = np.load(f"{dataset_path}/x.npy")
    y_all = np.load(f"{dataset_path}/y.npy")
    pos_all = np.load(f"{dataset_path}/pos.npy")
    cls_all = np.load(f"{dataset_path}/cls.npy")
else:
    X_all, y_all, pos_all, cls_all = extract_all_features(
        files, n_files, n_layers, n_heads, min_position, max_position
    )
    os.makedirs(dataset_path, exist_ok=True)
    np.save(f"{dataset_path}/x.npy", X_all)
    np.save(f"{dataset_path}/y.npy", y_all)
    np.save(f"{dataset_path}/pos.npy", pos_all)
    np.save(f"{dataset_path}/cls.npy", cls_all)

# ------------------ BALANCE FP SAMPLES ------------------

# Balance FP samples in training and testing datasets
X_train, y_train, pos_train, cls_train = balance_fp_samples(X_all[:n_train], y_all[:n_train], pos_all[:n_train], cls_all[:n_train], fp_replication_factor)
X_test, y_test, pos_test, cls_test = balance_fp_samples(X_all[n_train:], y_all[n_train:], pos_all[n_train:], cls_all[n_train:], fp_replication_factor)

# ------------------ SPLIT ------------------

n_total = len(X_all)
n_train = int(n_total * train_size)
X_train, X_test = X_all[:n_train], X_all[-(n_total - n_train):]
y_train, y_test = y_all[:n_train], y_all[-(n_total - n_train):]
pos_train, pos_test = pos_all[:n_train], pos_all[-(n_total - n_train):]
cls_train, cls_test = cls_all[:n_train], cls_all[-(n_total - n_train):]

# ------------------ STATS ------------------

unique, counts = np.unique(y_all, return_counts=True)
print("\nClass distribution (entire dataset):")
for u, c in zip(unique, counts):
    print(f"  Class {u}: {c} samples")

# ------------------ NORMALIZATION ------------------

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_test_t  = torch.tensor(y_test, dtype=torch.long)

if normalize_features:
    print("\nNormalizing features...")
    mean = X_train_t.mean(dim=0, keepdim=True)
    std = X_train_t.std(dim=0, keepdim=True) + 1e-6
    X_train_t = (X_train_t - mean) / std
    X_test_t = (X_test_t - mean) / std

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=128, shuffle=False)

# ------------------ MODEL ------------------

class MLPClassifierMulticlass(nn.Module):
    def __init__(self, input_dim, hidden1=512, hidden2=64, dropout_rate=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 3)  # 3 classes
        )

    def forward(self, x):
        return self.net(x)

input_dim = X_train_t.shape[1]
clf = MLPClassifierMulticlass(input_dim, dropout_rate=dropout_rate).to(device)

# ------------------ TRAINING ------------------

optimizer = optim.Adam(clf.parameters(), lr=1e-4, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

train_losses, val_losses = [], []
for epoch in range(n_epochs):
    clf.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = clf(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)

    # Validation
    clf.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = clf(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    val_loss = val_loss / len(test_loader.dataset)
    acc = correct / total
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1:02d}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.3f}")

# ------------------ EVALUATION ------------------

clf.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = clf(xb)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        y_pred.extend(preds)
        y_true.extend(yb.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

acc = accuracy_score(y_true, y_pred)
print(f"\nOverall Accuracy: {acc:.3f}")

# ------------------ CLASS STATS ------------------

labels = ["FP (0)", "TP (1)", "Other (2)"]
print("\nPer-class sample counts:")
for i, lbl in enumerate(labels):
    n_train_i = np.sum(y_train == i)
    n_test_i = np.sum(y_test == i)
    print(f"  {lbl}: Train={n_train_i}, Test={n_test_i}")

# ------------------ CONFUSION MATRIX ------------------

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("3×3 Confusion Matrix (Multiclass MLP)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(base_save_dir, "confusion_matrix_3x3.png"), dpi=130)
plt.show()

# ------------------ PRECISION/RECALL/F1 (FP vs Non-FP) ------------------

y_true_binary = (y_true == 0).astype(int)
y_pred_binary = (y_pred == 0).astype(int)

prec, rec, f1, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, average="binary")
print("\nBinary FP-vs-NonFP metrics:")
print(f"  Precision: {prec:.3f}")
print(f"  Recall:    {rec:.3f}")
print(f"  F1-score:  {f1:.3f}")

# ------------------ LOSS CURVE ------------------

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Multiclass MLP Training Loss")
plt.tight_layout()
plt.savefig(os.path.join(base_save_dir, "training_loss_curve.png"), dpi=130)
plt.close()

# ------------------ PERFORMANCE OVER TOKEN POSITION ------------------

token_positions = np.unique(pos_test)
accuracy_over_position = []

for pos in token_positions:
    indices = pos_test == pos
    acc = accuracy_score(y_test[indices], y_pred[indices])
    accuracy_over_position.append(acc)

plt.figure(figsize=(8, 5))
plt.plot(token_positions, accuracy_over_position, marker='o')
plt.xlabel("Token Position")
plt.ylabel("Accuracy")
plt.title("Classifier Performance over Token Positions")
plt.tight_layout()
plt.savefig(os.path.join(base_save_dir, "performance_over_positions.png"), dpi=130)
plt.show()

print(f"\nResults saved in:\n  {base_save_dir}")
