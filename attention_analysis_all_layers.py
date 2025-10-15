import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import seaborn as sns
from scipy.stats import sem

data_dir = "data/all layers all attention tp fp/a/"
files = glob(os.path.join(data_dir, "attentions_*.pkl"))
n_files = 3960
n_layers, n_heads = 15, 32
avg_win_size = 2
stride_size = 1
n_top_k = 5
n_subtokens = 1
offset_layer = 14
eps = 1e-8
sns.set(style="darkgrid")

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
            mean_vals = np.mean(topk_vals[..., : n_top_k], axis=-1)
            results.append((idx, mean_vals))
    return results

def aggregate_across_images(files, n_files):
    tp_collect = []
    fp_collect = []
    oth_collect = []
    for f in tqdm(files[:n_files]):
        try:
            with open(f, "rb") as handle:
                data_dict = pickle.load(handle)
        except Exception:
            continue
        tp_collect.extend(extract_attention_values(data_dict, "tp"))
        fp_collect.extend(extract_attention_values(data_dict, "fp"))
        oth_collect.extend(extract_attention_values(data_dict, "other"))
        
    return tp_collect, fp_collect, oth_collect

def aggregate_by_position(attention_data, n_layers, n_heads):
    layer_head_data = {l: {h: {} for h in range(n_heads)} for l in range(n_layers)}
    for idx, mean_vals in attention_data:
        for l in range(n_layers):
            for h in range(n_heads):
                layer_head_data[l][h].setdefault(int(idx), []).append(float(mean_vals[l, h]))
    return layer_head_data

def smooth_with_ci_from_posmap(pos_map, win=3, stride=1):
    if not pos_map:
        return np.array([]), np.array([]), np.array([])
    positions = sorted(pos_map.keys())
    x_arr = np.array(positions)
    xs, ys, cis = [], [], []
    for start in range(0, len(x_arr) - win + 1, stride):
        window_x = x_arr[start:start+win]
        min_x, max_x = int(window_x[0]), int(window_x[-1])
        window_vals = []
        for p in range(min_x, max_x + 1):
            if p in pos_map:
                window_vals.extend(pos_map[p])
        if not window_vals:
            continue
        xs.append(np.mean(window_x))
        ys.append(np.mean(window_vals))
        cis.append(1.96 * sem(window_vals) if len(window_vals) > 1 else 0.0)
    return np.array(xs), np.array(ys), np.array(cis)


def plot_attention_grid(tp_posmap, fp_posmap, oth_posmap, n_layers, n_heads,
                        savepath="attention_grid.pdf", x_min=9, x_max=141):
    fig_w = n_heads * 4
    fig_h = n_layers * 3
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(fig_w, fig_h), sharex=False, sharey=False)
    fig.suptitle("Attention to Image Tokens (TP: green, FP: red, Others: gray)", fontsize=60)
    
    for l in range(n_layers):
        for h in range(n_heads):
            ax = axes[l, h]
            tp_map = tp_posmap[l][h]
            fp_map = fp_posmap[l][h]
            oth_map = oth_posmap[l][h]
            tp_x, tp_y, tp_ci = smooth_with_ci_from_posmap(tp_map, win=avg_win_size, stride=stride_size)
            fp_x, fp_y, fp_ci = smooth_with_ci_from_posmap(fp_map, win=avg_win_size, stride=stride_size)
            oth_x, oth_y, oth_ci = smooth_with_ci_from_posmap(oth_map, win=avg_win_size, stride=stride_size)
            
            if tp_x.size > 0:
                ax.plot(tp_x, tp_y, color="tab:green", linewidth=1.4)
                ax.fill_between(tp_x, tp_y - tp_ci, tp_y + tp_ci, color="tab:blue", alpha=0.2)
            if fp_x.size > 0:
                ax.plot(fp_x, fp_y, color="tab:red", linewidth=1.4)
                ax.fill_between(fp_x, fp_y - fp_ci, fp_y + fp_ci, color="tab:red", alpha=0.2)
            if oth_x.size > 0:
                ax.plot(oth_x, oth_y, color="tab:gray", linewidth=1.4)
                ax.fill_between(oth_x, oth_y - oth_ci, oth_y + oth_ci, color="tab:gray", alpha=0.2)
                
            ax.set_xlim(x_min, x_max)
            ys_for_limits = []
            if tp_y.size > 0:
                ys_for_limits.extend((tp_y - 0.1*tp_ci)[6: -30].tolist())
                ys_for_limits.extend((tp_y + 0.1*tp_ci)[6: -30].tolist())
            if fp_y.size > 0:
                ys_for_limits.extend((fp_y - 0.1*fp_ci)[4: -30].tolist())
                ys_for_limits.extend((fp_y + 0.1*fp_ci)[4: -30].tolist())
            if ys_for_limits:
                y_min = min(ys_for_limits)
                y_max = max(ys_for_limits)
                if y_max == y_min:
                    y_min -= 1e-6
                    y_max += 1e-6
                y_pad = 0.05 * (y_max - y_min)
                ax.set_ylim(y_min - y_pad, y_max + y_pad)
            else:
                ax.set_ylim(0.0, 1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if l == n_layers - 1:
                ax.set_xlabel(f"H{h}", fontsize=30)
            if h == 0:
                ax.set_ylabel(f"L{l+offset_layer}", fontsize=30)
                
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(savepath)
    plt.show()

tp_data, fp_data, oth_data = aggregate_across_images(files, n_files)
tp_posmap = aggregate_by_position(tp_data, n_layers, n_heads)
fp_posmap = aggregate_by_position(fp_data, n_layers, n_heads)
oth_posmap = aggregate_by_position(oth_data, n_layers, n_heads)


plot_attention_grid(tp_posmap, fp_posmap, oth_posmap, n_layers, n_heads, savepath="attention_grid.pdf")
