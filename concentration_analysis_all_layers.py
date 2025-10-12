import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import seaborn as sns
from scipy.stats import sem

data_dir = "data/all layers attention tp fp"
files = glob(os.path.join(data_dir, "attentions_*.pkl"))
n_files = 3900
n_layers, n_heads = 15, 32
avg_win_size = 2
stride_size = 1
n_top_k = 20
n_subtokens = 1
offset_layer = 14
eps = 1e-8
sns.set(style="darkgrid")

concentration_metric = "std_spatial"  # options: 'entropy', 'inverse_simpson', 'gini', 'mass_ratio', 'std_spatial'
grid_size = 24

def compute_concentration(topk_indices, topk_values, metric):
    vals = np.maximum(topk_values, 0)
    if vals.sum() == 0:
        return np.nan
    p = vals / (vals.sum() + eps)
    if metric == "entropy":
        return -np.sum(p * np.log(p + eps))
    elif metric == "inverse_simpson":
        return 1.0 / np.sum(p ** 2)
    elif metric == "gini":
        sorted_p = np.sort(p)
        n = len(p)
        return 1 - 2 * np.sum((n - np.arange(1, n + 1) + 0.5) * sorted_p) / n
    elif metric == "mass_ratio":
        return np.sum(np.sort(p)[-5:])
    elif metric == "std_spatial":
        coords = np.array([[idx % grid_size, idx // grid_size] for idx in topk_indices])
        cx, cy = np.sum(p * coords[:, 0]), np.sum(p * coords[:, 1])
        dx = np.sqrt(np.sum(p * (coords[:, 0] - cx) ** 2))
        dy = np.sqrt(np.sum(p * (coords[:, 1] - cy) ** 2))
        return np.sqrt(dx ** 2 + dy ** 2)
    else:
        return np.nan

def extract_concentration_values(data_dict, cls_):
    results = []
    entries = data_dict.get(cls_, {}).get("image", [])
    for e in entries:
        if not e.get("subtoken_results"):
            continue
        for sub in e["subtoken_results"][:n_subtokens]:
            topk_vals = np.array(sub["topk_values"], dtype=float)
            topk_inds = np.array(sub["topk_indices"], dtype=int)
            if topk_vals.ndim != 3:
                continue
            idx = int(sub["idx"])
            metric_vals = np.zeros((n_layers, n_heads))
            for l in range(n_layers):
                for h in range(n_heads):
                    metric_vals[l, h] = compute_concentration(topk_inds[l, h, :n_top_k],
                                                              topk_vals[l, h, :n_top_k],
                                                              concentration_metric)
            results.append((idx, metric_vals))
    return results

def aggregate_across_images(files, n_files):
    tp_collect = []
    fp_collect = []
    for f in tqdm(files[:n_files]):
        try:
            with open(f, "rb") as handle:
                data_dict = pickle.load(handle)
        except Exception:
            continue
        tp_collect.extend(extract_concentration_values(data_dict, "tp"))
        fp_collect.extend(extract_concentration_values(data_dict, "fp"))
    return tp_collect, fp_collect

def aggregate_by_position(data, n_layers, n_heads):
    layer_head_data = {l: {h: {} for h in range(n_heads)} for l in range(n_layers)}
    for idx, vals in data:
        for l in range(n_layers):
            for h in range(n_heads):
                layer_head_data[l][h].setdefault(int(idx), []).append(float(vals[l, h]))
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

def plot_concentration_grid(tp_posmap, fp_posmap, n_layers, n_heads,
                            savepath, x_min=9, x_max=141):
    fig_w = n_heads * 1.0
    fig_h = n_layers * 0.8
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(fig_w, fig_h),
                             sharex=False, sharey=False)
    fig.suptitle(f"Concentration Metric: {concentration_metric} (TP: blue, FP: red)", fontsize=40)
    for l in range(n_layers):
        for h in range(n_heads):
            ax = axes[l, h]
            tp_map = tp_posmap[l][h]
            fp_map = fp_posmap[l][h]
            tp_x, tp_y, tp_ci = smooth_with_ci_from_posmap(tp_map, win=avg_win_size, stride=stride_size)
            fp_x, fp_y, fp_ci = smooth_with_ci_from_posmap(fp_map, win=avg_win_size, stride=stride_size)
            if tp_x.size > 0:
                ax.plot(tp_x, tp_y, color="tab:blue", linewidth=1.4)
                ax.fill_between(tp_x, tp_y - tp_ci, tp_y + tp_ci, color="tab:blue", alpha=0.2)
            if fp_x.size > 0:
                ax.plot(fp_x, fp_y, color="tab:red", linewidth=1.4)
                ax.fill_between(fp_x, fp_y - fp_ci, fp_y + fp_ci, color="tab:red", alpha=0.2)
            ax.set_xlim(x_min, x_max)
            ys_for_limits = []
            if tp_y.size > 0:
                ys_for_limits.extend((tp_y - tp_ci).tolist())
                ys_for_limits.extend((tp_y + tp_ci).tolist())
            if fp_y.size > 0:
                ys_for_limits.extend((fp_y - fp_ci).tolist())
                ys_for_limits.extend((fp_y + fp_ci).tolist())
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
                ax.set_xlabel(f"H{h}", fontsize=10)
            if h == 0:
                ax.set_ylabel(f"L{l+offset_layer}", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(savepath, dpi=300)
    plt.show()

tp_data, fp_data = aggregate_across_images(files, n_files)
tp_posmap = aggregate_by_position(tp_data, n_layers, n_heads)
fp_posmap = aggregate_by_position(fp_data, n_layers, n_heads)

save_name = f"concentration_grid_{concentration_metric}.pdf"
plot_concentration_grid(tp_posmap, fp_posmap, n_layers, n_heads, savepath=save_name)
