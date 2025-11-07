#!/usr/bin/env python3
# pope_two_classifiers.py
import os
import pickle
import numpy as np
import json
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

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

# -------------------------
# Repro
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -------------------------
# Paths / params
data_dir = "data/pope/"
attn_dir = os.path.join(data_dir, "pope_llava_attentions_greedy_all_layers_top20_adversarial")
resp_dir = os.path.join(data_dir, "pope_llava_responses_greedy_all_layers_top20_adversarial")
gt_file = f"{data_dir}coco/coco_pope_adversarial.json"

save_base = "results_pope_classifier_two_models"
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

# -------------------------
# Utilities: entropy / gini
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

# -------------------------
# Load GT and model responses
def load_ground_truths(gt_path):
    with open(gt_path, "r") as f:
        gt_data = [json.loads(line) for line in f]
    # map question_id -> label (lowercase)
    return {entry["question_id"]: entry["label"].lower() for entry in gt_data}

def load_responses(resp_dir):
    responses = {}
    for response_file in glob(os.path.join(resp_dir, "*.jsonl")):
        # Some files may have multiple lines but in your dataset there is one JSON object per file.
        try:
            with open(response_file, "r") as f:
                data = json.load(f)
        except Exception:
            # If the file contains lines, try to read first line
            with open(response_file, "r") as f:
                line = f.readline().strip()
                if not line:
                    continue
                data = json.loads(line)
        image_id = int(data["image_id"])
        question_id = int(data["question_id"])
        responses[(image_id, question_id)] = data["caption"].lower()
    return responses

# -------------------------
# Attention extraction (image attentions)
def extract_attention_features(data_dict):
    """
    Returns a list of arrays each shaped (n_layers, n_heads, n_top_k)
    For the POPE dataset each file usually contains one item.
    """
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

# -------------------------
# Build dataset: X, y (GT), cls labels, resp_bin
def extract_all_features(files):
    X, y, cls_list, resp_bins = [], [], [], []
    ground_truths = load_ground_truths(gt_file)
    responses = load_responses(resp_dir)

    for fpath in tqdm(files, desc="Extracting features"):
        try:
            with open(fpath, "rb") as handle:
                data_dict = pickle.load(handle)
        except Exception:
            # skip broken files silently
            continue

        fname = os.path.basename(fpath)
        # expected pattern: attentions_<imageid>_<questionid>.pkl
        parts = fname.split("_")
        if len(parts) < 3:
            # unexpected filename
            continue
        try:
            image_id = int(parts[1])
            question_id = int(parts[2].split(".")[0])
        except Exception:
            continue

        gt_label = ground_truths.get(question_id, None)
        resp_label = responses.get((image_id, question_id), None)
        if gt_label is None or resp_label is None:
            # not found, skip
            continue

        gt_label = gt_label.lower()
        resp_label = resp_label.lower()
        if gt_label not in ["yes", "no"] or resp_label not in ["yes", "no"]:
            continue

        gt_bin = 1 if gt_label == "yes" else 0
        resp_bin = 1 if resp_label == "yes" else 0

        attn_list = extract_attention_features(data_dict)
        if len(attn_list) == 0:
            continue

        # There may be multiple entries, iterate through them
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
            # NOTE: we do NOT append resp_bin to the feature vector.
            X.append(features)
            y.append(gt_bin)
            resp_bins.append(resp_bin)

            # also infer human-readable class for accounting (TP/FP/FN/TN)
            if resp_bin == 1 and gt_bin == 1:
                cls_list.append("TP")
            elif resp_bin == 1 and gt_bin == 0:
                cls_list.append("FP")
            elif resp_bin == 0 and gt_bin == 1:
                cls_list.append("FN")
            else:
                cls_list.append("TN")

    if len(X) == 0:
        return None, None, None, None

    return np.array(X), np.array(y), np.array(cls_list), np.array(resp_bins)

# -------------------------
# Balancing function (replicate FP and FN entries)
def balance_fp_fn_samples(X, y, cls_list, fp_factor=1, fn_factor=1):
    """
    Replicates FP and FN rows by specified factors.
    If there are no FP/FN rows, returns original arrays.
    """
    cls_arr = np.array(cls_list)
    # masks
    mask_fp = cls_arr == "FP"
    mask_fn = cls_arr == "FN"

    if mask_fp.sum() == 0 and mask_fn.sum() == 0:
        return X, y, cls_list

    parts_X = [X]
    parts_y = [y]
    parts_cls = [cls_arr]

    if mask_fp.sum() > 0 and fp_factor > 0:
        parts_X.append(np.repeat(X[mask_fp], fp_factor, axis=0))
        parts_y.append(np.repeat(y[mask_fp], fp_factor, axis=0))
        parts_cls.append(np.repeat(cls_arr[mask_fp], fp_factor, axis=0))

    if mask_fn.sum() > 0 and fn_factor > 0:
        parts_X.append(np.repeat(X[mask_fn], fn_factor, axis=0))
        parts_y.append(np.repeat(y[mask_fn], fn_factor, axis=0))
        parts_cls.append(np.repeat(cls_arr[mask_fn], fn_factor, axis=0))

    X_bal = np.concatenate(parts_X, axis=0)
    y_bal = np.concatenate(parts_y, axis=0)
    cls_bal = np.concatenate(parts_cls, axis=0)
    return X_bal, y_bal, cls_bal

# -------------------------
# MLP model
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

# -------------------------
# Train & evaluate helper (single subset)
def train_and_evaluate_classifier(X, y, cls_list, subset_name):
    """
    Trains one classifier on the subset X,y,cls_list.
    Saves model, scaler and confusion matrix under save_base/<subset_name>_*
    Returns (acc, precision, recall, f1)
    """
    os.makedirs(save_base, exist_ok=True)

    if X is None or len(X) == 0:
        print(f"No data for subset '{subset_name}', skipping.")
        return None

    # shuffle
    X, y, cls_list = shuffle(X, y, cls_list, random_state=SEED)

    # train / test split
    n_total = len(X)
    n_train = int(n_total * train_size)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    cls_train, cls_test = cls_list[:n_train], cls_list[n_train:]

    # balance (replicate FP/FN only)
    X_train, y_train, cls_train = balance_fp_fn_samples(X_train, y_train, cls_train,
                                                        fp_replication_factor, fn_replication_factor)
    X_test, y_test, cls_test = balance_fp_fn_samples(X_test, y_test, cls_test,
                                                     fp_replication_factor, fn_replication_factor)

    # convert to tensors and normalize (using train stats)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    mean = X_train_t.mean(dim=0, keepdim=True)
    std = X_train_t.std(dim=0, keepdim=True) + 1e-6
    X_train_t = (X_train_t - mean) / std
    X_test_t = (X_test_t - mean) / std
    torch.save({"mean": mean, "std": std}, os.path.join(save_base, f"{subset_name}_scaler.pt"))

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)

    input_dim = X_train_t.shape[1]
    clf = MLPClassifierTorch(input_dim, dropout_rate).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(clf.parameters(), lr=1e-3, weight_decay=weight_decay)

    train_losses, val_losses = [], []

    print(f"\nTraining classifier '{subset_name}' on {len(train_loader.dataset)} samples (train_size={train_size})...")
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

        # validation
        clf.eval()
        val_loss = 0.0
        correct, total = 0, 0
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
        acc = correct / total if total > 0 else float("nan")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"  Epoch {epoch:02d}/{n_epochs} | Train {train_loss:.4f} | Val {val_loss:.4f} | Val Acc {acc:.3f}")

    # save model
    model_path = os.path.join(save_base, f"{subset_name}_mlp.pt")
    torch.save(clf.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # final evaluation on test set
    clf.eval()
    y_pred_list, y_true_list = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = clf(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int).flatten()
            y_pred_list.extend(preds)
            y_true_list.extend(yb.cpu().numpy())

    y_pred = np.array(y_pred_list)
    y_true = np.array(y_true_list)
    if len(y_true) == 0:
        print(f"No test samples for {subset_name} after balancing — skipping metrics.")
        return None

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    print(f"\n=== Metrics for subset '{subset_name}' ===")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["GT=No", "GT=Yes"])
    fig = plt.figure(figsize=(5, 5))
    disp.plot(cmap="Blues", values_format="d", ax=fig.gca())
    plt.title(f"Confusion Matrix ({subset_name})")
    plt.tight_layout()
    cm_path = os.path.join(save_base, f"{subset_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=130)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    # save training curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title(f"Training Loss ({subset_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_base, f"{subset_name}_training_loss.png"), dpi=130)
    plt.close()

    return acc, precision, recall, f1

# -------------------------
# Main
def main():
    files = sorted(glob(os.path.join(attn_dir, "attentions_*.pkl")))
    print(f"Found {len(files)} attention files, extracting features ...")
    X_all, y_all, cls_all, resp_all = extract_all_features(files)
    if X_all is None:
        print("No features extracted. Exiting.")
        return

    print("Total samples:", len(X_all))
    # shuffle full dataset
    X_all, y_all, cls_all, resp_all = shuffle(X_all, y_all, cls_all, resp_all, random_state=SEED)

    # split by model response
    mask_yes = resp_all == 1
    mask_no = resp_all == 0

    X_yes, y_yes, cls_yes = X_all[mask_yes], y_all[mask_yes], cls_all[mask_yes]
    X_no, y_no, cls_no = X_all[mask_no], y_all[mask_no], cls_all[mask_no]

    print(f"Samples where model said YES: {len(X_yes)}")
    print(f"Samples where model said NO:  {len(X_no)}")

    results = {}
    if len(X_no) > 0:
        results["resp_no"] = train_and_evaluate_classifier(X_no, y_no, cls_no, subset_name="resp_no")
    else:
        print("No 'resp_no' samples to train on.")

    if len(X_yes) > 0:
        results["resp_yes"] = train_and_evaluate_classifier(X_yes, y_yes, cls_yes, subset_name="resp_yes")
    else:
        print("No 'resp_yes' samples to train on.")

    print("\nSummary:")
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
