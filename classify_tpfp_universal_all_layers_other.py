import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from scipy.stats import entropy as scipy_entropy

data_dir = "data/all layers all attention tp fp"
save_dir = "results_layerwise_headwise"
os.makedirs(save_dir, exist_ok=True)

n_files = 3900
n_layers, n_heads = 32, 32
min_position = 5
max_position = 150
position_margin = 2
n_top_k = 5
n_subtokens = 1
offset_layer = 14
eps = 1e-8
fp_replication_factor = 5  


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
    """Extract per-subtoken attention matrices."""
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
    """Aggregate mean attentions by position for each layer and head."""
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
                            entropy_vals = compute_entropy(attention_values)
                            gini_vals = compute_gini(attention_values)
                            features.extend([mean_attention, entropy_vals, gini_vals])
                if features:
                    pos_norm = token_pos / max_position
                    features.append(pos_norm)
                    X.append(features)
                    y.append(label)
                    pos_list.append(token_pos)
                    cls_list.append(cls_)
    if len(X) == 0:
        return None, None, None, None

    max_len = max(len(x) for x in X)
    X_padded = np.array([np.pad(x, (0, max_len - len(x)), constant_values=0) for x in X])
    y = np.array(y)
    pos_list = np.array(pos_list)
    cls_list = np.array(cls_list)
    return X_padded, y, pos_list, cls_list



files = sorted(glob(os.path.join(data_dir, "attentions_*.pkl")))

X_all, y_all, pos_all, cls_all = extract_all_features(
    files, n_files, n_layers, n_heads, min_position, max_position
)

split_idx = len(X_all) // 2
X_train, X_test = X_all[:split_idx], X_all[split_idx:]
y_train, y_test = y_all[:split_idx], y_all[split_idx:]
pos_train, pos_test = pos_all[:split_idx], pos_all[split_idx:]
cls_train, cls_test = cls_all[:split_idx], cls_all[split_idx:]

fp_mask = (y_train == 1)
X_fp, y_fp, pos_fp, cls_fp = X_train[fp_mask], y_train[fp_mask], pos_train[fp_mask], cls_train[fp_mask]
X_train_bal = np.concatenate([X_train, np.repeat(X_fp, fp_replication_factor, axis=0)], axis=0)
y_train_bal = np.concatenate([y_train, np.repeat(y_fp, fp_replication_factor, axis=0)], axis=0)
pos_train_bal = np.concatenate([pos_train, np.repeat(pos_fp, fp_replication_factor, axis=0)], axis=0)
cls_train_bal = np.concatenate([cls_train, np.repeat(cls_fp, fp_replication_factor, axis=0)], axis=0)

print(f"Balanced training size: {len(y_train_bal)} (FP={np.sum(y_train_bal==1)}, Non-FP={np.sum(y_train_bal==0)})")


clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
clf.fit(X_train_bal, y_train_bal)
y_pred = clf.predict(X_test)


precision_global, recall_global, f1_global, _ = precision_recall_fscore_support(
    y_test, y_pred, average='binary'
)
acc_global = accuracy_score(y_test, y_pred)

print("\n=== Global Metrics ===")
print(f"Precision: {precision_global:.3f}")
print(f"Recall:    {recall_global:.3f}")
print(f"F1-score:  {f1_global:.3f}")
print(f"Accuracy:  {acc_global:.3f}")


positions = np.arange(min_position, max_position + 1)
accs, precisions, recalls, f1s = [], [], [], []

for pos in positions:
    mask = np.abs(pos_test - pos) <= position_margin
    if np.sum(mask) == 0:
        accs.append(np.nan)
        precisions.append(np.nan)
        recalls.append(np.nan)
        f1s.append(np.nan)
        continue
    y_true_pos = y_test[mask]
    y_pred_pos = y_pred[mask]
    if len(np.unique(y_true_pos)) < 2:
        accs.append(np.nan)
        precisions.append(np.nan)
        recalls.append(np.nan)
        f1s.append(np.nan)
        continue
    n_pos_fp = np.sum(y_true_pos == 1)
    n_pos_nonfp = np.sum(y_true_pos == 0)
    if min(n_pos_fp, n_pos_nonfp) < 5:
        accs.append(np.nan)
        precisions.append(np.nan)
        recalls.append(np.nan)
        f1s.append(np.nan)
        continue
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_pos, y_pred_pos, average='binary')
    acc = accuracy_score(y_true_pos, y_pred_pos)
    accs.append(acc)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)


plt.figure(figsize=(10, 6))
plt.plot(positions, accs, label="Accuracy", linewidth=2)
plt.plot(positions, precisions, label="Precision", linewidth=2)
plt.plot(positions, recalls, label="Recall", linewidth=2)
plt.plot(positions, f1s, label="F1-score", linewidth=2)
plt.xlabel("Token Position", fontsize=14)
plt.ylabel("Metric Value", fontsize=14)
plt.title("Classifier Performance over Token Positions (FP vs Non-FP)", fontsize=15)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "position_metrics_FP_vs_nonFP.png"), dpi=130)
plt.show()


if hasattr(clf, "predict_proba"):
    probs = clf.predict_proba(X_test)[:, 1]
    fp_probs = probs[y_test == 1]
    tp_probs = probs[(y_test == 0) & (cls_test == "tp")]
    oth_probs = probs[(y_test == 0) & (cls_test == "other")]

    print("\n=== Mean predicted probability of FP (by class) ===")
    print(f"  FP:     {np.mean(fp_probs):.3f}")
    print(f"  TP:     {np.mean(tp_probs):.3f}")
    print(f"  Other:  {np.mean(oth_probs):.3f}")
