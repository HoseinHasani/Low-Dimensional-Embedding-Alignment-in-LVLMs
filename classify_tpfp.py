import os
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

data_dir = "data/attentions_temp1"
files = sorted(glob(os.path.join(data_dir, "attentions_*.pkl")))

selected_position = 47
position_margin = 2
n_train = 2000
n_test = 2000

X, y = [], []

for f in tqdm(files[:n_train + n_test]):
    with open(f, "rb") as handle:
        data_dict = pickle.load(handle)

    for cls_, label in [("tp", 1), ("fp", 0)]:
        for modality in ["image", "text"]:
            entries = data_dict.get(cls_, {}).get(modality, [])
            for e in entries:
                token_indices = e.get("token_indices", [])
                if len(token_indices) == 0 or len(e.get("subtoken_results", [])) == 0:
                    continue
                token_pos = int(token_indices[0])
                if abs(token_pos - selected_position) > position_margin:
                    continue
                subtoken_means = []
                for sub in e["subtoken_results"]:
                    if "topk_values" in sub and len(sub["topk_values"]) > 0:
                        subtoken_means.append(sub["topk_values"])
                if not subtoken_means:
                    continue
                subtoken_means = np.array(subtoken_means, dtype=float)
                features = subtoken_means.flatten()
                X.append(features)
                y.append(label)

max_len = max(len(x) for x in X)
X_padded = np.array([np.pad(x, (0, max_len - len(x)), constant_values=0) for x in X])
y = np.array(y)

X_train, X_test = X_padded[:n_train], X_padded[n_train:n_train + n_test]
y_train, y_test = y[:n_train], y[n_train:n_train + n_test]

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
acc = accuracy_score(y_test, y_pred)

print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")
print(f"Accuracy:  {acc:.3f}")
