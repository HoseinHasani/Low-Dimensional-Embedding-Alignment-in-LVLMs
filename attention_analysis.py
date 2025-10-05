import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from scipy.stats import sem

data_dir = "data/attentions_greedy"
files = glob(os.path.join(data_dir, "attentions_*.pkl"))

n_files = 4000
avg_win_size = 2
stride_size = 1

results = {"tp": {"image": [], "text": []},
            "fp": {"image": [], "text": []}}

for count, f in enumerate(files[:n_files]):
    with open(f, "rb") as handle:
        data_dict = pickle.load(handle)
    for cls in ["tp", "fp"]:
        for modality in ["image", "text"]:
            entries = data_dict.get(cls, {}).get(modality, [])
            for e in entries:
                if len(e["token_indices"]) == 0 or len(e["subtoken_results"]) == 0:
                    continue
                token_pos = e["token_indices"][0]
                subtoken_means = []
                for sub in e["subtoken_results"]:
                    if "topk_values" in sub and len(sub["topk_values"]) > 0:
                        subtoken_means.append(sub["topk_values"])
                if not subtoken_means:
                    continue
                subtoken_means = np.array(subtoken_means)
                mean_topk_values = np.mean(subtoken_means, axis=0)
                # attn_val = mean_topk_values[0] # max
                attn_val = np.mean(mean_topk_values) # mean
                
                results[cls][modality].append((token_pos, attn_val))

def aggregate_by_position(pairs):
    data = {}
    for pos, val in pairs:
        data.setdefault(pos, []).append(val)
    positions = sorted(data.keys())
    means = [np.mean(data[p]) for p in positions]
    return np.array(positions), np.array(means)

def smooth_with_ci(x, y, pairs, win=5, stride=1):
    if len(x) == 0:
        return np.array([]), np.array([]), np.array([])
    xs, ys, cis = [], [], []
    for start in range(0, len(x) - win + 1, stride):
        window_x = x[start:start+win]
        min_x, max_x = int(window_x[0]), int(window_x[-1])
        window_vals = [v for (p, v) in pairs if min_x <= p <= max_x]
        if not window_vals:
            continue
        xs.append(np.mean(window_x))
        ys.append(np.mean(window_vals))
        cis.append(1.96 * sem(window_vals) if len(window_vals) > 1 else 0)
    return np.array(xs), np.array(ys), np.array(cis)

def plot_modality(modality, results, avg_win_size, stride_size):
    plt.figure(figsize=(10, 6), facecolor="#f7f7f7")
    sns.set(style="whitegrid")

    colors = {"tp": "tab:blue", "fp": "tab:red"}

    for cls in ["tp", "fp"]:
        pairs = results[cls][modality]
        if not pairs:
            continue
        x, y = aggregate_by_position(pairs)
        x_smooth, y_smooth, y_ci = smooth_with_ci(x, y, pairs, avg_win_size, stride_size)
        plt.plot(x_smooth, y_smooth, color=colors[cls], label=f"{cls.upper()} - {modality}", linewidth=2)
        plt.fill_between(x_smooth, y_smooth - y_ci, y_smooth + y_ci, color=colors[cls], alpha=0.2)

    plt.xlabel("Token Position in Generated Text", fontsize=14)
    plt.ylabel("Average Attention Value", fontsize=14)
    plt.title(f"Attention to Previous Tokens ({modality.capitalize()} Modality)", fontsize=15)
    plt.xlim(-1, 150)
    plt.ylim(0.005 if modality == "image" else 0.21,
              0.03 if modality == "image" else 0.235)
    
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"attention_{modality}_tp_fp.png", dpi=140)
    plt.show()

plot_modality("image", results, avg_win_size, stride_size)
plot_modality("text", results, avg_win_size, stride_size)
