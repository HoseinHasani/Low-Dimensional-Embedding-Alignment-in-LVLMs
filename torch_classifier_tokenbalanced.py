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
from scipy.stats import entropy as scipy_entropy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict


# -----------------------------
# Configuration
# -----------------------------
data_dir = "data/all layers all attention tp fp"
base_save_dir = "tokenbalanced_results_all_layers"
os.makedirs(base_save_dir, exist_ok=True)

dataset_path = "cls_data__e_True_g_False"
use_text_attentions = True

n_files = 3900
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

n_epochs = 2
weight_decay = 1e-3
dropout_rate = 0.5

normalize_features = True
classifier_type = "pytorch_mlp"

exp_name = f"{classifier_type}_exp__ent{int(use_entropy)}_gin{int(use_gini)}"
save_dir = os.path.join(base_save_dir, exp_name)
model_dir = os.path.join(save_dir, "model")
results_dir = os.path.join(save_dir, "results")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")


# -----------------------------
# Helper Functions
# -----------------------------
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


def extract_attention_values(data_dict, cls_, source="image"):
    results = []
    entries = data_dict.get(cls_, {}).get(source, [])
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


def extract_all_features(files, n_files, n_layers, n_heads, min_position, max_position):
    X, y, pos_list, cls_list = [], [], [], []
    for f in tqdm(files[:n_files], desc="Extracting features"):
        try:
            with open(f, "rb") as handle:
                data_dict = pickle.load(handle)
        except Exception:
            continue

        for cls_, label in [("fp", 1), ("tp", 0), ("other", 0)]:
            img_samples = extract_attention_values(data_dict, cls_, "image")

            if use_text_attentions:
                txt_samples = extract_attention_values(data_dict, cls_, "text")
            else:
                txt_samples = []

            all_samples = []
            if img_samples:
                all_samples.extend([("image", *s) for s in img_samples])
            if use_text_attentions and txt_samples:
                all_samples.extend([("text", *s) for s in txt_samples])

            for src, idx, topk_arr in all_samples:
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

                if use_text_attentions:
                    features.append(1.0 if src == "text" else 0.0)

                features.append(token_pos / max_position)

                X.append(features)
                y.append(label)
                pos_list.append(token_pos)
                cls_list.append(cls_)

    if len(X) == 0:
        return None, None, None, None

    return np.array(X), np.array(y), np.array(pos_list), np.array(cls_list)


# -----------------------------
# Adaptive Replication Functions
# -----------------------------
def compute_adaptive_fp_replication_factors(y_all, pos_all, win=5):
    """Compute adaptive FP replication factor per token position."""
    min_pos, max_pos = int(pos_all.min()), int(pos_all.max())
    replication_factors = {}
    pos_to_labels = defaultdict(list)

    for pos, label in zip(pos_all, y_all):
        pos_to_labels[int(pos)].append(int(label))

    for j in range(min_pos, max_pos + 1):
        local_labels = []
        for k in range(j - win, j + win + 1):
            if k in pos_to_labels:
                local_labels.extend(pos_to_labels[k])

        if len(local_labels) == 0:
            replication_factors[j] = 1
            continue

        n_0 = np.sum(np.array(local_labels) == 0)
        n_1 = np.sum(np.array(local_labels) == 1)
        if n_1 == 0:
            replication_factors[j] = 1
        else:
            replication_factors[j] = max(int(np.round(n_0 / n_1)), 1)

    print(f"Computed adaptive replication factors for positions {min_pos}–{max_pos}")
    return replication_factors


def balance_fp_samples_adaptive(X, y, pos, cls, fp_factors):
    """Replicate FP samples based on adaptive per-position factors."""
    X_bal, y_bal, pos_bal, cls_bal = [X], [y], [pos], [cls]

    for p, factor in fp_factors.items():
        mask = (y == 1) & (pos == p)
        if np.any(mask) and factor > 1:
            X_rep = np.repeat(X[mask], factor, axis=0)
            y_rep = np.repeat(y[mask], factor, axis=0)
            pos_rep = np.repeat(pos[mask], factor, axis=0)
            cls_rep = np.repeat(cls[mask], factor, axis=0)
            X_bal.append(X_rep)
            y_bal.append(y_rep)
            pos_bal.append(pos_rep)
            cls_bal.append(cls_rep)

    X_bal = np.concatenate(X_bal, axis=0)
    y_bal = np.concatenate(y_bal, axis=0)
    pos_bal = np.concatenate(pos_bal, axis=0)
    cls_bal = np.concatenate(cls_bal, axis=0)

    return X_bal, y_bal, pos_bal, cls_bal


# -----------------------------
# Load or Extract Dataset
# -----------------------------
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

    if X_all is not None:
        dataset_path = f"cls_data__e_{use_entropy}_g_{use_gini}"
        os.makedirs(dataset_path, exist_ok=True)
        np.save(os.path.join(dataset_path, "x.npy"), X_all)
        np.save(os.path.join(dataset_path, "y.npy"), y_all)
        np.save(os.path.join(dataset_path, "pos.npy"), pos_all)
        np.save(os.path.join(dataset_path, "cls.npy"), cls_all)
        print(f"Dataset saved in '{dataset_path}/'")


# -----------------------------
# Train/Test Split
# -----------------------------
n_total = len(X_all)
n_train = int(n_total * train_size)
n_test = int(n_total * test_size)
if n_train + n_test > n_total:
    n_test = n_total - n_train

X_train, X_test = X_all[:n_train], X_all[-n_test:]
y_train, y_test = y_all[:n_train], y_all[-n_test:]
pos_train, pos_test = pos_all[:n_train], pos_all[-n_test:]
cls_train, cls_test = cls_all[:n_train], cls_all[-n_test:]


# -----------------------------
# Apply Adaptive FP Balancing
# -----------------------------
train_fp_factors = compute_adaptive_fp_replication_factors(y_all, pos_train, win=5)
test_fp_factors = train_fp_factors #compute_adaptive_fp_replication_factors(y_test, pos_test, win=5)

X_train, y_train, pos_train, cls_train = balance_fp_samples_adaptive(
    X_train, y_train, pos_train, cls_train, train_fp_factors
)
X_test, y_test, pos_test, cls_test = balance_fp_samples_adaptive(
    X_test, y_test, pos_test, cls_test, test_fp_factors
)

print(f"Train size: {len(y_train)} | FP={np.sum(y_train==1)}, Non-FP={np.sum(y_train==0)}")
print(f"Test size:  {len(y_test)} | FP={np.sum(y_test==1)}, Non-FP={np.sum(y_test==0)}")


# Optional: visualize replication factors
plt.figure(figsize=(10, 4))
plt.plot(list(train_fp_factors.keys()), list(train_fp_factors.values()), 'o-', label="Train factors")
plt.plot(list(test_fp_factors.keys()), list(test_fp_factors.values()), 'x--', label="Test factors")
plt.xlabel("Token Position")
plt.ylabel("Adaptive FP Replication Factor")
plt.title("Adaptive FP Replication per Token Position (win=5)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "adaptive_fp_replication_factors.png"), dpi=130)
plt.close()


# -----------------------------
# Convert to Torch
# -----------------------------
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

if normalize_features:
    print("\nNormalizing features (PyTorch)...")
    mean = X_train_t.mean(dim=0, keepdim=True)
    std = X_train_t.std(dim=0, keepdim=True) + 1e-6
    X_train_t = (X_train_t - mean) / std
    X_test_t = (X_test_t - mean) / std
    torch.save({"mean": mean, "std": std}, os.path.join(model_dir, "scaler.pt"))

train_ds = TensorDataset(X_train_t, y_train_t)
test_ds = TensorDataset(X_test_t, y_test_t)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)


# -----------------------------
# MLP Classifier
# -----------------------------
class MLPClassifierTorch(nn.Module):
    def __init__(self, input_dim, hidden1=512, hidden2=64, dropout_rate=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


input_dim = X_train_t.shape[1]
clf = MLPClassifierTorch(input_dim, dropout_rate=dropout_rate).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(clf.parameters(), lr=1e-3, weight_decay=weight_decay)


# -----------------------------
# Training Loop
# -----------------------------
train_losses, test_losses = [], []
print(f"\nTraining PyTorch MLP for {n_epochs} epochs...")

for epoch in range(n_epochs):
    clf.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.float().to(device)
        optimizer.zero_grad()
        logits = clf(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)

    clf.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.float().to(device)
            logits = clf(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item() * xb.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            correct += (preds == yb.long()).sum().item()
            total += yb.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    val_loss = val_loss / len(test_loader.dataset)
    acc = correct / total
    train_losses.append(train_loss)
    test_losses.append(val_loss)
    print(f"Epoch {epoch+1:02d}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.3f}")


torch.save(clf.state_dict(), os.path.join(model_dir, "pytorch_mlp_with_l2.pt"))
print(f"\nModel saved to: {os.path.join(model_dir, 'pytorch_mlp_with_l2.pt')}")


# -----------------------------
# Evaluation
# -----------------------------
clf.eval()
y_pred_list, y_true_list = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        yb = yb.float()
        logits = clf(xb)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int).flatten()
        y_pred_list.extend(preds)
        y_true_list.extend(yb.numpy())

y_pred = np.array(y_pred_list)
y_true = np.array(y_true_list)

precision_global, recall_global, f1_global, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
acc_global = accuracy_score(y_true, y_pred)

print("\n=== Global Metrics (PyTorch) ===")
print(f"Precision: {precision_global:.3f}")
print(f"Recall:    {recall_global:.3f}")
print(f"F1-score:  {f1_global:.3f}")
print(f"Accuracy:  {acc_global:.3f}")

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-FP", "FP"])
plt.figure(figsize=(5, 5))
disp.plot(cmap="Blues", values_format="d", colorbar=False)
plt.title("Confusion Matrix (PyTorch MLP)")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "confusion_matrix_pytorch.png"), dpi=130)
plt.close()


# FP / TP / Other confusion
conf_matrix = np.zeros((3, 2), dtype=int)
for true_label, predicted_label, cls in zip(y_true, y_pred, cls_test):
    if cls == 'fp':
        row = 0
    elif cls == 'tp':
        row = 1
    else:
        row = 2
    col = predicted_label
    conf_matrix[row, col] += 1

row_labels = ['FP', 'TP', 'Other']
col_labels = ['Class 0', 'Class 1']
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues",
            xticklabels=col_labels, yticklabels=row_labels, cbar=False)
plt.title("Confusion Matrix (FP, TP, Other vs. Class 0/1)", fontsize=14)
plt.xlabel('Predicted Labels', fontsize=12)
plt.ylabel('True Labels', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "confusion_matrix_fp_tp_other_seaborn.png"), dpi=130)
plt.close()


# Loss curves
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("PyTorch MLP Training Loss")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "training_loss_pytorch.png"), dpi=130)
plt.close()


# -----------------------------
# Positional Performance
# -----------------------------
positions = np.arange(min_position, max_position + 1)
accs, precisions, recalls, f1s = [], [], [], []

for pos in positions:
    mask = np.abs(pos_test - pos) <= position_margin
    if np.sum(mask) == 0:
        accs.append(np.nan); precisions.append(np.nan); recalls.append(np.nan); f1s.append(np.nan)
        continue
    y_true_pos, y_pred_pos = y_true[mask], y_pred[mask]
    accs.append(accuracy_score(y_true_pos, y_pred_pos))
    p, r, f, _ = precision_recall_fscore_support(y_true_pos, y_pred_pos, average='binary', zero_division=0)
    precisions.append(p); recalls.append(r); f1s.append(f)

plt.figure(figsize=(12, 6))
plt.plot(positions, accs, label="Accuracy")
plt.plot(positions, precisions, label="Precision")
plt.plot(positions, recalls, label="Recall")
plt.plot(positions, f1s, label="F1-score")
plt.xlabel("Token Position")
plt.ylabel("Score")
plt.legend()
plt.title("Metrics by Token Position (PyTorch MLP)")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "metrics_by_position_pytorch.png"), dpi=130)
plt.close()
