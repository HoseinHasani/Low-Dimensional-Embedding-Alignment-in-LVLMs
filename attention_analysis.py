import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from scipy.stats import sem

data_dir = "data/attentions"
files = glob(os.path.join(data_dir, "attentions_*.pkl"))

results = {"tp": {"image": [], "text": []},
            "fp": {"image": [], "text": []}}

for f in files:
    with open(f, "rb") as handle:
        data_dict = pickle.load(handle)

    for cls in ["tp", "fp"]:
        for modality in ["image", "text"]:
            entries = data_dict.get(cls, {}).get(modality, [])
            for e in entries:
                if len(e["token_indices"]) == 0 or len(e["mean_topk_values"]) == 0:
                    continue  
                token_pos = e["token_indices"][0]
                attn_val = np.mean(e["mean_topk_values"])
                results[cls][modality].append((token_pos, attn_val))

def to_arrays(pairs):
    if not pairs:
        return np.array([]), np.array([])
    pairs = sorted(pairs, key=lambda x: x[0])
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    return x, y

def aggregate_by_position(pairs):
    data = {}
    for pos, val in pairs:
        data.setdefault(pos, []).append(val)
    positions = sorted(data.keys())
    means = [np.mean(data[p]) for p in positions]
    cis = [1.96 * sem(data[p]) if len(data[p]) > 1 else 0 for p in positions]
    return np.array(positions), np.array(means), np.array(cis)

plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")

colors = {"tp": "tab:blue", "fp": "tab:red"}

for cls in ["tp", "fp"]:
    for modality, linestyle in zip(["image", "text"], ["-", "--"]):
        pairs = results[cls][modality]
        if not pairs:
            continue
        x, y_mean, y_ci = aggregate_by_position(pairs)
        plt.plot(x, y_mean, linestyle, color=colors[cls], label=f"{cls.upper()} - {modality}")
        plt.fill_between(x, y_mean - y_ci, y_mean + y_ci, color=colors[cls], alpha=0.2)

plt.xlabel("Token Position in Generated Text")
plt.ylabel("Average Attention Value")
plt.title("Attention to Previous Tokens: TP vs FP (Image vs Text)")
plt.legend()
plt.tight_layout()
plt.show()
