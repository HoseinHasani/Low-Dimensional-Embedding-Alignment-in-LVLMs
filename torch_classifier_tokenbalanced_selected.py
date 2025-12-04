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
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score
)
from scipy.stats import entropy as scipy_entropy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict


sns.set_style('darkgrid')
# sns.set_palette('bright')

# -----------------------------
# Configuration
# -----------------------------
data_dir = "data/all layers all attention tp fp"
base_save_dir = "final_cl_results"
os.makedirs(base_save_dir, exist_ok=True)
dataset_path = "cls_data"

#######################
n_select_att = 15     # number of best attention heads
n_select_ent = 3     # number of best entropy heads

zero_class = 'tp'
balanced_train = False
balanced_test = False
fp2tp_ratio = 0.7
test_thresh = 0.5
n_files = 3900
train_size = 0.75
test_size = 0.25
###################

if zero_class == 'tp':
    other_dropout = 0.9
    tp_dropout = 0.0
elif zero_class == 'other':
    other_dropout = 0.0
    tp_dropout = 0.9
else:
    other_dropout = 0.2
    tp_dropout = 0.0    

use_text_attentions = False
n_layers, n_heads = 32, 32
min_position = 5
max_position = 155
position_margin = 2
n_top_k = 20
n_subtokens = 1
eps = 1e-10
use_entropy = True
n_epochs = 2
weight_decay = 1e-3
dropout_rate = 0.005
normalize_features = True

exp_name = f"exp__a{n_select_att}_e{n_select_ent}_trainb_{int(balanced_train)}_testb_{int(balanced_test)}_zero_{zero_class}"
save_dir = os.path.join(base_save_dir, exp_name)
model_dir = os.path.join(save_dir, "model")
results_dir = os.path.join(save_dir, "results")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")


# =========================
# Load aggregated statistics
# =========================

agg_att = np.load("stats_summary/attention_aggregated.npy", allow_pickle=True).item()
agg_ent = np.load("stats_summary/entropy_aggregated.npy", allow_pickle=True).item()


# Convert dict keys "L_i_H_j" → tuple (i,j)
def parse_key(k):
    parts = k.split("_")
    return int(parts[1]), int(parts[3])

# Sort by ascending score (because minimum-pooled values: small = more distinguishable)
sorted_att = sorted(agg_att.items(), key=lambda x: x[1])
sorted_ent = sorted(agg_ent.items(), key=lambda x: x[1])

best_att_heads = [parse_key(k) for k, v in sorted_att[:n_select_att]]
best_ent_heads = [parse_key(k) for k, v in sorted_ent[:n_select_ent]]

print(f"Selected {len(best_att_heads)} attention heads")
print(f"Selected {len(best_ent_heads)} entropy heads")





# -----------------------------
# Helper Functions
# -----------------------------
def compute_entropy(values):
    if len(values) == 0:
        return 0.0
    prob = np.array(values, dtype=float)
    prob = prob / (np.sum(prob, axis=-1, keepdims=True) + eps)
    return -np.sum(prob * np.log(prob + eps), axis=-1)


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
            txt_samples = extract_attention_values(data_dict, cls_, "text") if use_text_attentions else []
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

                
                # Extract only selected attention heads
                for (l, h) in best_att_heads:
                    vals = topk_arr[l, h, :]
                    features.append(np.mean(vals))     # attention mean
                
                # Extract only selected entropy heads
                if use_entropy:
                    for (l, h) in best_ent_heads:
                        vals = topk_arr[l, h, :]
                        features.append(compute_entropy(vals))
        
        
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
            replication_factors[j] = max(int(fp2tp_ratio * np.round(n_0 / n_1)), 1)

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

def drop_samples(X, y, pos, cls, target="other", dropout_ratio=0.5):
    """
    Randomly remove a fraction of 'target' class samples from the dataset.
    """
    mask_target = (cls == target)
    target_indices = np.where(mask_target)[0]

    if dropout_ratio <= 0 or len(target_indices) == 0:
        return X, y, pos, cls
    if dropout_ratio >= 1.0:
        keep_mask = ~mask_target
    else:
        n_drop = int(len(target_indices) * dropout_ratio)
        drop_indices = np.random.choice(target_indices, size=n_drop, replace=False)
        keep_mask = np.ones(len(cls), dtype=bool)
        keep_mask[drop_indices] = False

    print(f"Dropped {np.sum(~keep_mask)} / {len(cls)} ('{target}' samples removed: {100*dropout_ratio:.1f}%)")
    return X[keep_mask], y[keep_mask], pos[keep_mask], cls[keep_mask]



# -----------------------------
# Load or Extract Dataset
# -----------------------------
files = sorted(glob(os.path.join(data_dir, "attentions_*.pkl")))

# if dataset_path and os.path.exists(f"{dataset_path}/x.npy"):
if False:
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
        dataset_path = "cls_data"
        os.makedirs(dataset_path, exist_ok=True)
        # np.save(os.path.join(dataset_path, "x.npy"), X_all)
        # np.save(os.path.join(dataset_path, "y.npy"), y_all)
        # np.save(os.path.join(dataset_path, "pos.npy"), pos_all)
        # np.save(os.path.join(dataset_path, "cls.npy"), cls_all)
        # print(f"Dataset saved in '{dataset_path}/'")
        

print('X.shape', X_all.shape)

if other_dropout > 0:
    X_all, y_all, pos_all, cls_all = drop_samples(
        X_all, y_all, pos_all, cls_all, target="other", dropout_ratio=other_dropout
    )

if tp_dropout > 0:
    X_all, y_all, pos_all, cls_all = drop_samples(
        X_all, y_all, pos_all, cls_all, target="tp", dropout_ratio=other_dropout
    )
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


if balanced_train:
    X_train, y_train, pos_train, cls_train = balance_fp_samples_adaptive(
        X_train, y_train, pos_train, cls_train, train_fp_factors
    )

if balanced_test:
    X_test, y_test, pos_test, cls_test = balance_fp_samples_adaptive(
        X_test, y_test, pos_test, cls_test, test_fp_factors
    )




print(f"Train size: {len(y_train)} | FP={np.sum(y_train==1)}, Non-FP={np.sum(y_train==0)}")
print(f"Test size:  {len(y_test)} | FP={np.sum(y_test==1)}, Non-FP={np.sum(y_test==0)}")


# -----------------------------
# Torch Model
# -----------------------------
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)
if normalize_features:
    mean, std = X_train_t.mean(0, keepdim=True), X_train_t.std(0, keepdim=True) + 1e-6
    X_train_t, X_test_t = (X_train_t - mean) / std, (X_test_t - mean) / std
    torch.save({"mean": mean, "std": std}, os.path.join(model_dir, "scaler.pt"))

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=128, shuffle=False)

class MLPClassifierTorch(nn.Module):
    def __init__(self, input_dim, hidden1=512, hidden2=64, dropout_rate=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, hidden1), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

clf = MLPClassifierTorch(X_train_t.shape[1], dropout_rate=dropout_rate).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(clf.parameters(), lr=1e-3, weight_decay=weight_decay)

train_losses, test_losses = [], []
print(f"\nTraining PyTorch MLP for {n_epochs} epochs...")
for epoch in range(n_epochs):
    clf.train(); running_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.float().to(device)
        optimizer.zero_grad()
        loss = criterion(clf(xb), yb)
        loss.backward(); optimizer.step()
        running_loss += loss.item() * xb.size(0)
    clf.eval(); val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.float().to(device)
            logits = clf(xb)
            loss = criterion(logits, yb)
            probs = torch.sigmoid(logits)
            preds = (probs > test_thresh).long()
            correct += (preds.cpu() == yb.cpu().long()).sum().item()
            val_loss += loss.item() * xb.size(0)
            total += yb.size(0)
    train_losses.append(running_loss / len(train_loader.dataset))
    test_losses.append(val_loss / len(test_loader.dataset))
    print(f"Epoch {epoch+1}/{n_epochs} | Val Acc: {correct/total:.3f}")

torch.save(clf.state_dict(), os.path.join(model_dir, "pytorch_mlp.pt"))
print("Model saved.\n")

# -----------------------------
# Evaluation (with probabilities)
# -----------------------------
clf.eval()
y_probs, y_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        probs = torch.sigmoid(clf(xb)).cpu().numpy()
        y_probs.extend(probs); y_true.extend(yb.numpy())
y_probs = np.array(y_probs).flatten()
y_pred = (y_probs > test_thresh).astype(int)
y_true = np.array(y_true)

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
acc = accuracy_score(y_true, y_pred)
metrics_text = f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | Acc: {acc:.3f}"
print(metrics_text)

metrics_file = os.path.join(results_dir, "global_metrics.txt")
with open(metrics_file, 'w') as f:
    f.write(metrics_text)
    
# -----------------------------
# ROC + PR Curves
# -----------------------------
fpr, tpr, _ = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)
precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
ap_score = average_precision_score(y_true, y_probs)

plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.3f}")
plt.plot([0,1],[0,1],'--',c='gray')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve", fontweight='bold'); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(results_dir, "roc_curve.png"), dpi=300); plt.close()

plt.figure(figsize=(6,5))
plt.plot(recall_curve, precision_curve, lw=2, label=f"AP={ap_score:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision–Recall Curve"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(results_dir, "precision_recall_curve.png"), dpi=300); plt.close()

# -----------------------------
# Normalized Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)
cm_percent = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_percent*100, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=["Non-FP","FP"], yticklabels=["Non-FP","FP"],
            annot_kws={"size":12,"weight":"bold"}, cbar=False)
plt.title("Normalized Confusion Matrix (%)", fontweight='bold'); plt.tight_layout()
plt.savefig(os.path.join(results_dir, "confusion_matrix_percent.png"), dpi=300); plt.close()


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

# Compute percentages (row-wise)
conf_percent = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
conf_percent = np.nan_to_num(conf_percent)  # avoid NaN if a row sums to zero

# Combine counts + percentages in display text
annot_text = np.empty_like(conf_matrix, dtype=object)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        count = conf_matrix[i, j]
        pct = conf_percent[i, j] * 100
        annot_text[i, j] = f"{pct:.1f}%"

# Plot heatmap
row_labels = ['FP', 'TP', 'Other']
col_labels = ['Class 0', 'Class 1']

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=annot_text, fmt='', cmap="Blues",
            xticklabels=col_labels, yticklabels=row_labels, cbar=False,
            annot_kws={"fontsize": 10, "ha": "center", "va": "center"})

plt.title("Confusion Matrix (FP, TP, Other vs. Predicted Labels)", fontsize=11, fontweight='bold')
plt.xlabel('Predicted Labels', fontsize=12)
plt.ylabel('True Labels', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "confusion_matrix_fp_tp_other_with_percentages.png"), dpi=300)
plt.close()


# -----------------------------
# Per-token metrics (with AUROC)
# -----------------------------

# Smoothing kernel
kernel = np.array([0.05, 0.15, 0.6, 0.15, 0.05])


def smooth(arr, kernel):

    sm = arr.copy()

    for j in range(2, len(arr)-2):
        sm[j] = np.sum(arr[j-2:j+3]*kernel)

    return sm

positions = np.arange(min_position, max_position+1)
accs, precs, recs, f1s, aurocs = [], [], [], [], []
for pos in positions:
    mask = np.abs(pos_test - pos) <= position_margin
    if np.sum(mask) == 0 or len(np.unique(y_true[mask])) < 2:
        accs.append(np.nan)
        precs.append(np.nan)
        recs.append(np.nan)
        f1s.append(np.nan)
        aurocs.append(np.nan)
        continue
    y_t, y_p, y_pr = y_true[mask], y_pred[mask], y_probs[mask]
    accs.append(accuracy_score(y_t, y_p))
    p, r, f, _ = precision_recall_fscore_support(y_t, y_p, average='binary', zero_division=0)
    precs.append(p)
    recs.append(r)
    f1s.append(f)
    aurocs.append(roc_auc_score(y_t, y_pr))

# Apply smoothing
accs_smoothed = smooth(np.array(accs), kernel)
precs_smoothed = smooth(np.array(precs), kernel)
recs_smoothed = smooth(np.array(recs), kernel)
f1s_smoothed = smooth(np.array(f1s), kernel)
aurocs_smoothed = smooth(np.array(aurocs), kernel)

# Plotting the results
fig, ax = plt.subplots(figsize=(8, 5))

# Plot all metrics on the same axis
ax.plot(positions, accs_smoothed, label="Accuracy", linewidth=2)
ax.plot(positions, precs_smoothed, label="Precision", linewidth=2)
ax.plot(positions, recs_smoothed, label="Recall", linewidth=2)
ax.plot(positions, f1s_smoothed, label="F1", linewidth=2)
ax.plot(positions, aurocs_smoothed, label="AUROC", linewidth=2, linestyle='--', color='k')

np.save(os.path.join(results_dir, "accs.npy"), accs)
np.save(os.path.join(results_dir, "precs.npy"), precs)
np.save(os.path.join(results_dir, "recs.npy"), recs)
np.save(os.path.join(results_dir, "f1s.npy"), f1s)
np.save(os.path.join(results_dir, "aurocs.npy"), aurocs_smoothed)


plt.legend()
ax.set_xlabel("Token Position")
ax.set_ylabel("Metric Value")
plt.xlim(8, 151)
plt.title("Performance by Token Position", fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "metrics_by_position_with_auroc.png"), dpi=300)
plt.close()
