import os
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

data_dir = "data/attentions_greedy"
files = sorted(glob(os.path.join(data_dir, "attentions_*.pkl")))

n_files = 8000
position_margin = 2
min_position = 5
max_position = 150

train_files = files[: n_files // 2]
test_files = files[n_files // 2 : n_files]

def extract_all_features(files, position_margin, max_position):
    X, y = [], []
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
                    subtoken_means = []
                    for sub in subs[:1]:
                        topk = sub.get("topk_values", [])
                        if topk and isinstance(topk, (list, np.ndarray)):
                            subtoken_means.append(np.array(topk, dtype=float))
                    if not subtoken_means:
                        continue
                    subtoken_means = np.array(subtoken_means)
                    features = subtoken_means.flatten()
                    pos_norm = 2 * token_pos / max_position - 1
                    features = np.append(features, pos_norm)
                    X.append(features)
                    y.append(label)
    if len(X) == 0:
        return None, None
    max_len = max(len(x) for x in X)
    X_padded = np.array([np.pad(x, (0, max_len - len(x)), constant_values=0) for x in X])
    y = np.array(y)
    return X_padded, y

X_train, y_train = extract_all_features(train_files, position_margin, max_position)
X_test, y_test = extract_all_features(test_files, position_margin, max_position)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
acc = accuracy_score(y_test, y_pred)

print(f"Samples: Train={len(X_train)}  Test={len(X_test)}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")
print(f"Accuracy:  {acc:.3f}")
