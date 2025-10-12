import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import seaborn as sns

data_dir = "data/attentions"
files = glob(os.path.join(data_dir, "attentions_*.pkl"))

n_layers, n_heads = 15, 32
sns.set(style="darkgrid")

def extract_attention_values(data_dict, cls_):
    """Return list of tuples: (idx, mean_attention[layer, head])"""
    results = []
    entries = data_dict.get(cls_, {}).get("image", [])
    for e in entries:
        if not e.get("subtoken_results"):
            continue
        for sub in e["subtoken_results"]:
            topk_vals = np.array(sub["topk_values"], dtype=float)  
            if topk_vals.ndim != 3:
                continue
            idx = sub["idx"]  

            mean_vals = np.mean(topk_vals, axis=-1)
            results.append((idx, mean_vals))
    return results

def aggregate_across_images(files, n_files=200):
    tp_collect = []
    fp_collect = []
    for f in tqdm(files[:n_files]):
        with open(f, "rb") as handle:
            data_dict = pickle.load(handle)
        tp_collect.extend(extract_attention_values(data_dict, "tp"))
        fp_collect.extend(extract_attention_values(data_dict, "fp"))
    return tp_collect, fp_collect

tp_data, fp_data = aggregate_across_images(files, n_files=500)

def aggregate_by_position(attention_data, n_layers, n_heads):
    """Return dict[layer][head] = (positions, mean_values)"""
    layer_head_data = {l: {h: {} for h in range(n_heads)} for l in range(n_layers)}
    for idx, mean_vals in attention_data:
        for l in range(n_layers):
            for h in range(n_heads):
                layer_head_data[l][h].setdefault(idx, []).append(mean_vals[l, h])

    for l in range(n_layers):
        for h in range(n_heads):
            positions = sorted(layer_head_data[l][h].keys())
            if not positions:
                layer_head_data[l][h] = ([], [])
                continue
            means = [np.mean(layer_head_data[l][h][p]) for p in positions]
            layer_head_data[l][h] = (positions, means)
    return layer_head_data

tp_agg = aggregate_by_position(tp_data, n_layers, n_heads)
fp_agg = aggregate_by_position(fp_data, n_layers, n_heads)

def plot_attention_grid(tp_agg, fp_agg, n_layers, n_heads, savepath="attention_grid.png"):
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(32, 15), sharex=True, sharey=True)
    fig.suptitle("Attention to Image Tokens (TP: blue, FP: red)", fontsize=16)
    for l in range(n_layers):
        for h in range(n_heads):
            ax = axes[l, h]
            tp_x, tp_y = tp_agg[l][h]
            fp_x, fp_y = fp_agg[l][h]
            if len(tp_x) > 0:
                ax.plot(tp_x, tp_y, color="tab:blue", linewidth=1)
            if len(fp_x) > 0:
                ax.plot(fp_x, fp_y, color="tab:red", linewidth=1, linestyle="--")
            ax.set_xticks([])
            ax.set_yticks([])
            if l == n_layers - 1:
                ax.set_xlabel(f"H{h}", fontsize=6)
            if h == 0:
                ax.set_ylabel(f"L{l}", fontsize=6)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(savepath, dpi=200)
    plt.show()

plot_attention_grid(tp_agg, fp_agg, n_layers, n_heads, savepath="attention_grid_tp_fp.pdf")
