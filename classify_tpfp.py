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
n_files = 8000

files = sorted(glob(os.path.join(data_dir, "attentions_*.pkl")))
train_files = files[: n_files // 2]
test_files = files[n_files // 2 : n_files]

def extract_features(files, selected_position, position_margin):
    X, y = [], []
    for f in files:
        with open(f, "rb") as handle:
            data_dict = pickle.load(handle)
        for cls_, label in [("tp", 1), ("fp", 0)]:
            for modality in ["image"]:
                entries = data_dict.get(cls_, {}).get(modality, [])
                for e in entries:
                    token_indices = e.get("token_indices", [])
                    sub_results = e.get("subtoken_results", [])
                    if len(token_indices) == 0 or len(sub_results) == 0:
                        continue
                    token_pos = int(token_indices[0])
                    if abs(token_pos - selected_position) > position_margin:
                        continue
                    subtoken_means = []
                    for sub in sub_results[:1]:
                        topk = sub.get("topk_values", [])
                        if topk and isinstance(topk, (list, np.ndarray)):
                            subtoken_means.append(np.array(topk, dtype=float))
                    if not subtoken_means:
                        continue
                    subtoken_means = np.array(subtoken_means)
                    features = subtoken_means.flatten()
                    X.append(features)
                    y.append(label)
    if len(X) == 0:
        return None, None
    max_len = max(len(x) for x in X)
    X_padded = np.array([np.pad(x, (0, max_len - len(x)), constant_values=0) for x in X])
    y = np.array(y)
    return X_padded, y

positions, accs, precisions, recalls, f1s = [], [], [], [], []

for pos in tqdm(range(min_position, max_position + 1)):
    X_train, y_train = extract_features(train_files, pos, position_margin)
    X_test, y_test = extract_features(test_files, pos, position_margin)

    if X_train is None or X_test is None:
        continue
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        continue
    n_train_tp = np.sum(y_train == 1)
    n_train_fp = np.sum(y_train == 0)
    n_test_tp = np.sum(y_test == 1)
    n_test_fp = np.sum(y_test == 0)
    if min(n_train_tp, n_train_fp, n_test_tp, n_test_fp) < min_class_samples:
        continue

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    acc = accuracy_score(y_test, y_pred)

    positions.append(pos)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    accs.append(acc)

if len(positions) == 0:
    raise ValueError("No valid positions with sufficient samples were found.")

plt.figure(figsize=(10, 6))
plt.plot(positions, accs, label="Accuracy", linewidth=2)
plt.plot(positions, precisions, label="Precision", linewidth=2)
plt.plot(positions, recalls, label="Recall", linewidth=2)
plt.plot(positions, f1s, label="F1-score", linewidth=2)
plt.xlabel("Token Position", fontsize=14)
plt.ylabel("Metric Value", fontsize=14)
plt.title("Classification Performance over Token Positions", fontsize=15)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "positional_classification_metrics.png"), dpi=130)
plt.show()

results = {
    "positions": positions,
    "accuracy": accs,
    "precision": precisions,
    "recall": recalls,
    "f1": f1s
}

# np.savez(os.path.join(save_dir, "positional_metrics.npz"), **results)
