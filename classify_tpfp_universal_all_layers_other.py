import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from scipy.stats import entropy as scipy_entropy
import joblib

data_dir = "data/all layers all attention tp fp"
base_save_dir = "results_layerwise_headwise"
os.makedirs(base_save_dir, exist_ok=True)

dataset_path = "cls_data"

n_files = 3900
n_layers, n_heads = 32, 32
min_position = 5
max_position = 150
position_margin = 2
n_top_k = 20
n_subtokens = 1
eps = 1e-10
fp_replication_factor = 10

use_entropy = True
use_gini = True

train_size = 0.7   
test_size  = 0.3   

normalize_features = True
classifier_type = "mlp"  # choose 'rf' or 'mlp'
# -------------------------------------------------

exp_name = f"{classifier_type}_exp__ent{int(use_entropy)}_gin{int(use_gini)}"
save_dir = os.path.join(base_save_dir, exp_name)
model_dir = os.path.join(save_dir, "model")
results_dir = os.path.join(save_dir, "results")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

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
            mean_vals = np.mean(topk_vals[..., :n_top_k], axis=-1)
            results.append((idx, mean_vals))
    return results

def aggregate_by_position(attention_data, n_layers, n_heads):
    layer_head_data = {l: {h: {} for h in range(n_heads)} for l in range(n_layers)}
    for idx, mean_vals in attention_data:
        for l in range(n_layers):
            for h in range(n_heads):
                layer_head_data[l][h].setdefault(int(idx), []).append(float(mean_vals[l, h]))
    return layer_head_data

def extract_all_features(files, n_files, n_layers, n_heads, min_position, max_position):
    X, y, pos_list, cls_list = [], [], [], []
    for f in tqdm(files[:n_files], desc="Extracting features"):
        try:
            with open(f, "rb") as handle:
                data_dict = pickle.load(handle)
        except Exception:
            continue

        for cls_, label in [("fp", 1), ("tp", 0), ("other", 0)]:
            attention_data = extract_attention_values(data_dict, cls_)
            layer_head_data = aggregate_by_position(attention_data, n_layers, n_heads)
            for idx, mean_vals in attention_data:
                token_pos = int(idx)
                if token_pos < min_position or token_pos > max_position:
                    continue
                features = []
                for l in range(n_layers):
                    for h in range(n_heads):
                        attention_values = layer_head_data[l][h].get(token_pos, [])
                        if attention_values:
                            mean_attention = np.mean(attention_values)
                            features.append(mean_attention)
                            if use_entropy:
                                entropy_vals = compute_entropy(attention_values)
                                features.append(entropy_vals)
                            if use_gini:
                                gini_vals = compute_gini(attention_values)
                                features.append(gini_vals)
                if features:
                    features.append(token_pos / max_position)
                    X.append(features)
                    y.append(label)
                    pos_list.append(token_pos)
                    cls_list.append(cls_)
    if len(X) == 0:
        return None, None, None, None

    max_len = max(len(x) for x in X)
    X_padded = np.array([np.pad(x, (0, max_len - len(x)), constant_values=0) for x in X])
    return X_padded, np.array(y), np.array(pos_list), np.array(cls_list)


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

n_total = len(X_all)
n_train = int(n_total * train_size)
n_test  = int(n_total * test_size)
if n_train + n_test > n_total:
    n_test = n_total - n_train

X_train, X_test = X_all[:n_train], X_all[-n_test:]
y_train, y_test = y_all[:n_train], y_all[-n_test:]
pos_train, pos_test = pos_all[:n_train], pos_all[-n_test:]
cls_train, cls_test = cls_all[:n_train], cls_all[-n_test:]

def balance_fp_samples(X, y, pos, cls, factor=5):
    fp_mask = (y == 1)
    X_fp, y_fp, pos_fp, cls_fp = X[fp_mask], y[fp_mask], pos[fp_mask], cls[fp_mask]
    X_bal = np.concatenate([X, np.repeat(X_fp, factor, axis=0)], axis=0)
    y_bal = np.concatenate([y, np.repeat(y_fp, factor, axis=0)], axis=0)
    pos_bal = np.concatenate([pos, np.repeat(pos_fp, factor, axis=0)], axis=0)
    cls_bal = np.concatenate([cls, np.repeat(cls_fp, factor, axis=0)], axis=0)
    return X_bal, y_bal, pos_bal, cls_bal

X_train, y_train, pos_train, cls_train = balance_fp_samples(X_train, y_train, pos_train, cls_train, fp_replication_factor)
X_test, y_test, pos_test, cls_test = balance_fp_samples(X_test, y_test, pos_test, cls_test, fp_replication_factor)

print(f"Train size: {len(y_train)} | FP={np.sum(y_train==1)}, Non-FP={np.sum(y_train==0)}")
print(f"Test size:  {len(y_test)} | FP={np.sum(y_test==1)}, Non-FP={np.sum(y_test==0)}")

scaler = None
if normalize_features:
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

if classifier_type == "rf":
    clf = RandomForestClassifier(n_estimators=200, random_state=7, class_weight="balanced")
elif classifier_type == "mlp":
    clf = MLPClassifier(
        hidden_layer_sizes=(512, 64),
        batch_size=64,
        activation="relu",
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=2,
        early_stopping=True,
        random_state=7,
        verbose=True,
    )
else:
    raise ValueError("Invalid classifier_type. Choose 'rf' or 'mlp'.")

print(f"\nTraining {classifier_type.upper()} classifier...")
clf.fit(X_train, y_train)
joblib.dump(clf, os.path.join(model_dir, f"{classifier_type}_model.pkl"))


y_pred = clf.predict(X_test)

precision_global, recall_global, f1_global, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
acc_global = accuracy_score(y_test, y_pred)

print("\n=== Global Metrics ===")
print(f"Precision: {precision_global:.3f}")
print(f"Recall:    {recall_global:.3f}")
print(f"F1-score:  {f1_global:.3f}")
print(f"Accuracy:  {acc_global:.3f}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-FP", "FP"])
plt.figure(figsize=(5, 5))
disp.plot(cmap="Blues", values_format="d", colorbar=False)
plt.title(f"Confusion Matrix ({classifier_type.upper()} Classifier)")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, f"confusion_matrix_{classifier_type}.png"), dpi=130)
plt.close()

positions = np.arange(min_position, max_position + 1)
accs, precisions, recalls, f1s = [], [], [], []

for pos in positions:
    mask = np.abs(pos_test - pos) <= position_margin
    if np.sum(mask) == 0:
        accs.append(np.nan); precisions.append(np.nan); recalls.append(np.nan); f1s.append(np.nan)
        continue
    y_true_pos, y_pred_pos = y_test[mask], y_pred[mask]
    if len(np.unique(y_true_pos)) < 2:
        accs.append(np.nan); precisions.append(np.nan); recalls.append(np.nan); f1s.append(np.nan)
        continue
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_pos, y_pred_pos, average='binary')
    acc = accuracy_score(y_true_pos, y_pred_pos)
    accs.append(acc); precisions.append(precision); recalls.append(recall); f1s.append(f1)

plt.figure(figsize=(10, 6))
plt.plot(positions, accs, label="Accuracy", linewidth=2)
plt.plot(positions, precisions, label="Precision", linewidth=2)
plt.plot(positions, recalls, label="Recall", linewidth=2)
plt.plot(positions, f1s, label="F1-score", linewidth=2)
plt.xlabel("Token Position", fontsize=14)
plt.ylabel("Metric Value", fontsize=14)
plt.title(f"Classifier Performance over Token Positions ({classifier_type.upper()})", fontsize=15)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, f"position_metrics_{classifier_type}.png"), dpi=130)
plt.close()

print(f"\nResults saved in:\n  {save_dir}")
