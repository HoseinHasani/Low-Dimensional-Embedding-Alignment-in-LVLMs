import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

data_dir = "data/attentions_greedy"
save_dir = "results_position_sweep"
os.makedirs(save_dir, exist_ok=True)

min_position = 5
max_position = 150
position_margin = 2
min_class_samples = 5
n_files = 3800

files = sorted(glob(os.path.join(data_dir, "attentions_*.pkl")))
train_files = files[: n_files // 2]
test_files = files[n_files // 2 : n_files]

def compute_entropy(values):
    vals = np.array(values, dtype=float)
    if vals.ndim != 3: return None
    probs = vals / (np.sum(vals, axis=-1, keepdims=True) + 1e-8)
    ent = -np.sum(probs * np.log(probs + 1e-8), axis=-1)
    return ent

def compute_gini(values):
    vals = np.array(values, dtype=float)
    if vals.ndim != 3: return None
    probs = vals / (np.sum(vals, axis=-1, keepdims=True) + 1e-8)
    sorted_p = np.sort(probs, axis=-1)
    n = sorted_p.shape[-1]
    coef = 2 * np.arange(1, n + 1) - n - 1
    gini = np.sum(coef * sorted_p, axis=-1) / (n * np.sum(sorted_p, axis=-1) + 1e-8)
    return np.abs(gini)

def extract_all_features(files, max_position, position_margin):
    X, y, pos_list = [], [], []
    for f in tqdm(files):
        with open(f, "rb") as handle:
            data_dict = pickle.load(handle)
        for cls_, label in [("tp", 1), ("fp", 0)]:
            for modality in ["image"]:
                entries = data_dict.get(cls_, {}).get(modality, [])
                for e in entries:
                    token_indices = e.get("token_indices", [])
                    subs = e.get("subtoken_results", [])
                    if len(token_indices) == 0 or len(subs) == 0:
                        continue
                    token_pos = int(token_indices[0])
                    if token_pos < min_position or token_pos > max_position:
                        continue
                    features = []
                    for sub in subs[:1]:  
                        topk = sub.get("topk_values", [])
                        if topk and isinstance(topk, (list, np.ndarray)):
                            mean_attention = np.mean(topk, axis=-1)
                            entropy = compute_entropy(topk)
                            gini = compute_gini(topk)
                            if mean_attention is not None and entropy is not None and gini is not None:
                                features.extend(mean_attention.flatten())
                                features.extend(entropy.flatten())
                                features.extend(gini.flatten())
                    if features:
                        pos_norm = token_pos / max_position
                        features = np.append(features, pos_norm)
                        X.append(features)
                        y.append(label)
                        pos_list.append(token_pos)
    if len(X) == 0:
        return None, None, None
    max_len = max(len(x) for x in X)
    X_padded = np.array([np.pad(x, (0, max_len - len(x)), constant_values=0) for x in X])
    y = np.array(y)
    pos_list = np.array(pos_list)
    return X_padded, y, pos_list

X_train, y_train, pos_train = extract_all_features(train_files, max_position, position_margin)
X_test, y_test, pos_test = extract_all_features(test_files, max_position, position_margin)

clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

precision_global, recall_global, f1_global, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
acc_global = accuracy_score(y_test, y_pred)

print(f"Global Metrics:")
print(f"Precision: {precision_global:.3f}")
print(f"Recall:    {recall_global:.3f}")
print(f"F1-score:  {f1_global:.3f}")
print(f"Accuracy:  {acc_global:.3f}")

# Token-based metrics
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
    n_pos_tp = np.sum(y_true_pos == 1)
    n_pos_fp = np.sum(y_true_pos == 0)
    if min(n_pos_tp, n_pos_fp) < min_class_samples:
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
plt.title("Conditional Classifier Performance over Token Positions", fontsize=15)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "conditional_classifier_position_metrics.png"), dpi=130)
plt.show()

