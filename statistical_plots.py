import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import seaborn as sns
from scipy.stats import sem, ttest_ind

data_dir = "data/all layers attention tp fp"
files = glob(os.path.join(data_dir, "attentions_*.pkl"))
save_dir = "selected_plots"
os.makedirs(save_dir, exist_ok=True)

n_files = 3900
n_layers, n_heads = 15, 32
avg_win_size = 7
stride_size = 1
n_top_k = 20
n_subtokens = 1
offset_layer = 14
eps = 1e-8
grid_size = 24
selected_positions = [30, 60, 80, 100, 130]
n_select = 30
sns.set(style="darkgrid")

def compute_entropy(values):
    vals = np.array(values, dtype=float)
    if vals.ndim != 3: return None
    probs = vals / (np.sum(vals, axis=-1, keepdims=True) + eps)
    ent = -np.sum(probs * np.log(probs + eps), axis=-1)
    return ent

def compute_gini(values):
    vals = np.array(values, dtype=float)
    if vals.ndim != 3: return None
    probs = vals / (np.sum(vals, axis=-1, keepdims=True) + eps)
    sorted_p = np.sort(probs, axis=-1)
    n = sorted_p.shape[-1]
    coef = 2 * np.arange(1, n + 1) - n - 1
    gini = np.sum(coef * sorted_p, axis=-1) / (n * np.sum(sorted_p, axis=-1) + eps)
    return np.abs(gini)

def extract_attention_values(data_dict, cls_):
    results = []
    entries = data_dict.get(cls_, {}).get("image", [])
    for e in entries:
        if not e.get("subtoken_results"): continue
        for sub in e["subtoken_results"][:n_subtokens]:
            vals = np.array(sub["topk_values"], dtype=float)
            if vals.ndim != 3: continue
            idx = int(sub["idx"])
            mean_vals = np.mean(vals[..., :n_top_k], axis=-1)
            ent_vals = compute_entropy(vals[..., :n_top_k])
            gini_vals = compute_gini(vals[..., :n_top_k])
            results.append((idx, mean_vals, ent_vals, gini_vals))
    return results

def aggregate_across_images(files, n_files):
    tp_collect, fp_collect = [], []
    for f in tqdm(files[:n_files]):
        try:
            with open(f, "rb") as handle:
                data_dict = pickle.load(handle)
        except Exception:
            continue
        tp_collect.extend(extract_attention_values(data_dict, "tp"))
        fp_collect.extend(extract_attention_values(data_dict, "fp"))
    return tp_collect, fp_collect

def aggregate_by_position(data, metric_id):
    layer_head_data = {l: {h: {} for h in range(n_heads)} for l in range(n_layers)}
    for idx, att, ent, gin in data:
        metric = [att, ent, gin][metric_id]
        if metric is None: continue
        for l in range(n_layers):
            for h in range(n_heads):
                layer_head_data[l][h].setdefault(int(idx), []).append(float(metric[l, h]))
    return layer_head_data

def smooth_with_ci_from_posmap(pos_map, win=3, stride=1):
    if not pos_map: return np.array([]), np.array([]), np.array([])
    positions = sorted(pos_map.keys())
    x_arr = np.array(positions)
    xs, ys, cis = [], [], []
    for start in range(0, len(x_arr) - win + 1, stride):
        window_x = x_arr[start:start+win]
        min_x, max_x = int(window_x[0]), int(window_x[-1])
        window_vals = []
        for p in range(min_x, max_x + 1):
            if p in pos_map: window_vals.extend(pos_map[p])
        if not window_vals: continue
        xs.append(np.mean(window_x))
        ys.append(np.mean(window_vals))
        cis.append(1.96 * sem(window_vals) if len(window_vals) > 1 else 0.0)
    return np.array(xs), np.array(ys), np.array(cis)

def plot_selected(tp_posmap, fp_posmap, metric_name, selected, save_dir, x_min=9, x_max=141):
    for (l, h, pval_log) in selected:
        tp_map = tp_posmap[l][h]
        fp_map = fp_posmap[l][h]
        tp_x, tp_y, tp_ci = smooth_with_ci_from_posmap(tp_map, win=avg_win_size, stride=stride_size)
        fp_x, fp_y, fp_ci = smooth_with_ci_from_posmap(fp_map, win=avg_win_size, stride=stride_size)
        plt.figure(figsize=(6,4))
        if tp_x.size > 0:
            plt.plot(tp_x, tp_y, color="tab:blue", linewidth=1.4)
            plt.fill_between(tp_x, tp_y - tp_ci, tp_y + tp_ci, color="tab:blue", alpha=0.2)
        if fp_x.size > 0:
            plt.plot(fp_x, fp_y, color="tab:red", linewidth=1.4)
            plt.fill_between(fp_x, fp_y - fp_ci, fp_y + fp_ci, color="tab:red", alpha=0.2)
        plt.xlim(x_min, x_max)
        ys_for_limits = []
        if tp_y.size > 0:
            ys_for_limits.extend((tp_y - 0.1*tp_ci)[6: -30].tolist())
            ys_for_limits.extend((tp_y + 0.1*tp_ci)[6: -30].tolist())
        if fp_y.size > 0:
            ys_for_limits.extend((fp_y - 0.1*fp_ci)[4: -30].tolist())
            ys_for_limits.extend((fp_y + 0.1*fp_ci)[4: -30].tolist())
        if ys_for_limits:
            y_min, y_max = min(ys_for_limits), max(ys_for_limits)
            if y_max == y_min:
                y_min -= 1e-6
                y_max += 1e-6
            y_pad = 0.05 * (y_max - y_min)
            plt.ylim(y_min - y_pad, y_max + y_pad)
        else:
            plt.ylim(0.0, 1.0)
        plt.title(f"{metric_name} L{l+offset_layer}H{h} log(p)={pval_log:.2f}", fontsize=12)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{metric_name}_L{l}_H{h}_logp{pval_log:.2f}.png"), dpi=130)
        plt.close()

tp_data, fp_data = aggregate_across_images(files, n_files)
tp_maps = [aggregate_by_position(tp_data, i) for i in range(3)]
fp_maps = [aggregate_by_position(fp_data, i) for i in range(3)]
metric_names = ["attention", "entropy", "gini"]

for midx, metric in enumerate(metric_names):
    pvals = []
    for l in range(n_layers):
        for h in range(n_heads):
            tp_map = tp_maps[midx][l][h]
            fp_map = fp_maps[midx][l][h]
            tp_vals, fp_vals = [], []
            for pos in selected_positions:
                tp_vals.extend(tp_map.get(pos, []))
                fp_vals.extend(fp_map.get(pos, []))
            if len(tp_vals) > 1 and len(fp_vals) > 1:
                _, p = ttest_ind(tp_vals, fp_vals, equal_var=False)
                pvals.append((l, h, p))
    pvals = sorted(pvals, key=lambda x: x[2])[:n_select]
    selected = [(l, h, -np.log10(p+eps)) for l, h, p in pvals]
    plot_selected(tp_maps[midx], fp_maps[midx], metric, selected, save_dir)
