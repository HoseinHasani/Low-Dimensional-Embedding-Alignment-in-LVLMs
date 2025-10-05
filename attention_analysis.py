import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from scipy.stats import sem
from tqdm import tqdm

data_dir = "data/attentions_temp1"
files = glob(os.path.join(data_dir, "attentions_*.pkl"))

n_files = 4000
avg_win_size = 3
stride_size = 1
eps = 1e-8

results = {"tp": {"image": [], "text": []},
           "fp": {"image": [], "text": []}}

sns.set(style="darkgrid")

for f in tqdm(files[:n_files]):
    with open(f, "rb") as handle:
        data_dict = pickle.load(handle)
    for cls_ in ["tp", "fp"]:
        for modality in ["image", "text"]:
            entries = data_dict.get(cls_, {}).get(modality, [])
            for e in entries:
                if len(e.get("token_indices", [])) == 0 or len(e.get("subtoken_results", [])) == 0:
                    continue
                token_pos = int(e["token_indices"][0])
                subtoken_means = []
                for sub in e["subtoken_results"]:
                    if "topk_values" in sub and len(sub["topk_values"]) > 0:
                        subtoken_means.append(sub["topk_values"])
                if not subtoken_means:
                    continue
                subtoken_means = np.array(subtoken_means, dtype=float)
                mean_topk_values = np.mean(subtoken_means, axis=0)
                attn_val = float(np.mean(mean_topk_values))
                results[cls_][modality].append((token_pos, attn_val))

def aggregate_by_position(pairs):
    data = {}
    for pos, val in pairs:
        data.setdefault(int(pos), []).append(float(val))
    positions = sorted(data.keys())
    means = [np.mean(data[p]) for p in positions]
    return np.array(positions), np.array(means), data

def smooth_with_ci(x, pairs, win=5, stride=1):
    if len(x) == 0:
        return np.array([]), np.array([]), np.array([])
    xs, ys, cis = [], [], []
    for start in range(0, len(x) - win + 1, stride):
        window_x = x[start:start+win]
        min_x, max_x = int(window_x[0]), int(window_x[-1])
        window_vals = [v for (p, v) in pairs if min_x <= int(p) <= max_x]
        if not window_vals:
            continue
        xs.append(np.mean(window_x))
        ys.append(np.mean(window_vals))
        cis.append(1.96 * sem(window_vals) if len(window_vals) > 1 else 0.0)
    return np.array(xs), np.array(ys), np.array(cis)

def plot_modality(modality, results, avg_win_size, stride_size):
    plt.figure(figsize=(9, 6))
    colors = {"tp": "tab:blue", "fp": "tab:red"}
    for cls_ in ["tp", "fp"]:
        pairs = results[cls_][modality]
        if not pairs:
            continue
        x, y, _ = aggregate_by_position(pairs)
        x_smooth, y_smooth, y_ci = smooth_with_ci(x, pairs, avg_win_size, stride_size)
        if x_smooth.size == 0:
            continue
        linestyle = "-" if modality == "image" else "--"
        plt.plot(x_smooth, y_smooth, color=colors[cls_], label=f"{cls_.upper()} - {modality}", linewidth=2, linestyle=linestyle)
        plt.fill_between(x_smooth, y_smooth - y_ci, y_smooth + y_ci, color=colors[cls_], alpha=0.2)
    plt.xlabel("Token Position in Generated Text", fontsize=14)
    plt.ylabel("Average Attention Value", fontsize=14)
    plt.title(f"Attention to Previous Tokens ({modality.capitalize()} Modality)", fontsize=15)
    plt.ylim(0.9 * np.min(y_smooth), 1.1 * np.max(y_smooth))
    plt.xlim(-1, 150)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"attention_{modality}_tp_fp.png", dpi=140)
    plt.show()

def plot_ratio(results, avg_win_size, stride_size, eps=1e-8):
    ratios = {"tp": [], "fp": []}
    for cls_ in ["tp", "fp"]:
        img_pairs = results[cls_]["image"]
        txt_pairs = results[cls_]["text"]
        txt_map = {}
        for p, v in txt_pairs:
            txt_map.setdefault(int(p), []).append(float(v))
        for p, img_val in img_pairs:
            p_int = int(p)
            txt_vals = txt_map.get(p_int)
            if not txt_vals:
                continue
            txt_mean = float(np.mean(txt_vals))
            ratio = float(img_val) / (txt_mean + eps)
            ratios[cls_].append((p_int, ratio))
    plt.figure(figsize=(9, 6))
    colors = {"tp": "tab:blue", "fp": "tab:red"}
    for cls_ in ["tp", "fp"]:
        pairs = ratios[cls_]
        if not pairs:
            continue
        x, y, _ = aggregate_by_position(pairs)
        x_smooth, y_smooth, y_ci = smooth_with_ci(x, pairs, avg_win_size, stride_size)
        if x_smooth.size == 0:
            continue
        plt.plot(x_smooth, y_smooth, color=colors[cls_], label=f"{cls_.upper()} ratio", linewidth=2)
        plt.fill_between(x_smooth, y_smooth - y_ci, y_smooth + y_ci, color=colors[cls_], alpha=0.2)
    plt.xlabel("Token Position in Generated Text", fontsize=14)
    plt.ylabel("Image/Text Attention Ratio", fontsize=14)
    plt.title("Image-to-Text Attention Ratio (TP vs FP)", fontsize=15)
    plt.ylim(0.9 * np.min(y_smooth), 1.1 * np.max(y_smooth))
    plt.xlim(-1, 150)
    plt.tight_layout()
    plt.legend(fontsize=12)
    plt.savefig("attention_ratio_tp_fp.png", dpi=140)
    plt.show()

plot_modality("image", results, avg_win_size, stride_size)
plot_modality("text", results, avg_win_size, stride_size)
plot_ratio(results, avg_win_size, stride_size, eps)
