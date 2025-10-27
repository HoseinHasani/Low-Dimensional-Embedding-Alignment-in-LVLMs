import os
import pickle
import numpy as np
import json
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

np.random.seed(0)

data_dir = "data/pope/"
attn_dir = os.path.join(data_dir, "pope_llava_attentions_greedy_all_layers_top20_adversarial")
resp_dir = os.path.join(data_dir, "pope_llava_responses_greedy_all_layers_top20_adversarial")
gt_file = f"{data_dir}coco/coco_pope_adversarial.json"

save_base = "results_pope_classifier"
os.makedirs(save_base, exist_ok=True)

n_layers, n_heads = 32, 32
n_top_k = 20
use_entropy = True
use_gini = False

fp_replication_factor = 6
fn_replication_factor = 2
train_size = 0.5
n_epochs = 3
weight_decay = 1e-3
dropout_rate = 0.5
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
eps = 1e-10


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


def load_ground_truths():
    with open(gt_file, "r") as f:
        gt_data = [json.loads(line) for line in f]
    return {entry["question_id"]: entry["label"].lower() for entry in gt_data}


def load_responses():
    responses = {}
    for response_file in glob(os.path.join(resp_dir, "*.jsonl")):
        with open(response_file, "r") as f:
            data = json.load(f)
            image_id = int(data["image_id"])
            question_id = int(data["question_id"])
            responses[(image_id, question_id)] = data["caption"].lower()
    return responses


def extract_attention_features(data_dict):
    results = []
    entries = data_dict.get("attns", {}).get("image", [])
    for e in entries:
        if not e.get("subtoken_results"):
            continue
        sub = e["subtoken_results"][0]
        topk_vals = np.array(sub["topk_values"], dtype=float)
        if topk_vals.ndim != 3:
            continue
        results.append(topk_vals[..., :n_top_k])
    return results


def extract_all_features(files):
    X, y, cls_list = [], [], []
    ground_truths = load_ground_truths()
    responses = load_responses()

    for f in tqdm(files, desc="Extracting features"):
        try:
            with open(f, "rb") as handle:
                data_dict = pickle.load(handle)
        except Exception:
            continue

        fname = os.path.basename(f)
        image_id = int(fname.split("_")[1])
        question_id = int(fname.split("_")[2].split(".")[0])

        gt_label = ground_truths.get(question_id, None)
        resp_label = responses.get((image_id, question_id), None)
        if not gt_label or not resp_label:
            print("WRONG FORMAT 1")
            continue

        gt_label = gt_label.lower()
        resp_label = resp_label.lower()
        if gt_label not in ["yes", "no"] or resp_label not in ["yes", "no"]:
            print("WRONG FORMAT 2:", gt_label, resp_label)
            continue

        gt_bin = 1 if gt_label == "yes" else 0
        resp_bin = 1 if resp_label == "yes" else 0

        attn_list = extract_attention_features(data_dict)
        if len(attn_list) == 0:
            continue

        for attn_vals in attn_list:
            features = []
            for l in range(n_layers):
                for h in range(n_heads):
                    vals = attn_vals[l, h, :]
                    mean_attention = np.mean(vals)
                    features.append(mean_attention)
                    if use_entropy:
                        features.append(compute_entropy(vals))
                    if use_gini:
                        features.append(compute_gini(vals))

            features.append(resp_bin)
            X.append(features)
            y.append(gt_bin)

            if resp_bin == 1 and gt_bin == 1:
                cls_list.append("TP")
            elif resp_bin == 1 and gt_bin == 0:
                cls_list.append("FP")
            elif resp_bin == 0 and gt_bin == 1:
                cls_list.append("FN")
            else:
                cls_list.append("TN")

    return np.array(X), np.array(y), np.array(cls_list)


def balance_fp_fn_samples(X, y, cls_list, fp_factor, fn_factor):
    cls_list = np.array(cls_list)
    mask_fp = cls_list == "FP"
    mask_fn = cls_list == "FN"

    X_fp, y_fp, cls_fp = X[mask_fp], y[mask_fp], cls_list[mask_fp]
    X_fn, y_fn, cls_fn = X[mask_fn], y[mask_fn], cls_list[mask_fn]

    X_bal = np.concatenate(
        [X, np.repeat(X_fp, fp_factor, axis=0), np.repeat(X_fn, fn_factor, axis=0)], axis=0
    )
    y_bal = np.concatenate(
        [y, np.repeat(y_fp, fp_factor, axis=0), np.repeat(y_fn, fn_factor, axis=0)], axis=0
    )
    cls_bal = np.concatenate(
        [cls_list, np.repeat(cls_fp, fp_factor, axis=0), np.repeat(cls_fn, fn_factor, axis=0)], axis=0
    )

    return X_bal, y_bal, cls_bal


class MLPClassifierTorch(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


files = sorted(glob(os.path.join(attn_dir, "attentions_*.pkl")))
X_all, y_all, cls_all = extract_all_features(files)

print(X_all.shape)

X_all, y_all, cls_all = shuffle(X_all, y_all, cls_all, random_state=42)

n_total = len(X_all)
n_train = int(n_total * train_size)
X_train, X_test = X_all[:n_train], X_all[n_train:]
y_train, y_test = y_all[:n_train], y_all[n_train:]
cls_train, cls_test = cls_all[:n_train], cls_all[n_train:]

X_train, y_train, cls_train = balance_fp_fn_samples(
    X_train, y_train, cls_train, fp_replication_factor, fn_replication_factor
)
X_test, y_test, cls_test = balance_fp_fn_samples(
    X_test, y_test, cls_test, fp_replication_factor, fn_replication_factor
)

print("\nDataset sizes after balancing:")
print(f"  Train: {len(y_train)}  |  Test: {len(y_test)}")
for name, cls_split in zip(["Train", "Test"], [cls_train, cls_test]):
    unique, counts = np.unique(cls_split, return_counts=True)
    print(f"{name} composition:")
    for u, c in zip(unique, counts):
        print(f"  {u}: {c}")


X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

mean = X_train_t.mean(dim=0, keepdim=True)
std = X_train_t.std(dim=0, keepdim=True) + 1e-6
X_train_t = (X_train_t - mean) / std
X_test_t = (X_test_t - mean) / std
torch.save({"mean": mean, "std": std}, os.path.join(save_base, "scaler.pt"))

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)

input_dim = X_train_t.shape[1]
clf = MLPClassifierTorch(input_dim, dropout_rate).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(clf.parameters(), lr=1e-3, weight_decay=weight_decay)

train_losses, val_losses = [], []
print(f"\nTraining on {len(train_loader.dataset)} samples...")

for epoch in range(1, n_epochs + 1):
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

    clf.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = clf(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds.cpu() == yb.cpu().long()).sum().item()
            total += yb.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    val_loss = val_loss / len(test_loader.dataset)
    acc = correct / total
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch:02d}/{n_epochs} | Train {train_loss:.4f} | Val {val_loss:.4f} | Acc {acc:.3f}")

torch.save(clf.state_dict(), os.path.join(save_base, "pytorch_mlp_pope.pt"))

clf.eval()
y_pred_list, y_true_list = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = clf(xb)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int).flatten()
        y_pred_list.extend(preds)
        y_true_list.extend(yb.numpy())

y_pred = np.array(y_pred_list)
y_true = np.array(y_true_list)

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
acc = accuracy_score(y_true, y_pred)

print("\n=== Global Metrics ===")
print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["GT=No", "GT=Yes"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (POPE Classifier)")
plt.tight_layout()
plt.savefig(os.path.join(save_base, "confusion_matrix.png"), dpi=130)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Loss (POPE MLP)")
plt.tight_layout()
plt.savefig(os.path.join(save_base, "training_loss.png"), dpi=130)
plt.close()

print(f"\nResults saved in '{save_base}/'")
