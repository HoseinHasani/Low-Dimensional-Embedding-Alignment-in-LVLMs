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
n_files = 1900
n_layers, n_heads = 15, 32
avg_win_size = 2
stride_size = 1
n_top_k = 20
n_subtokens = 1
offset_layer = 14
eps = 1e-8
grid_size = 24
selected_positions = [30, 60, 80, 100, 130]
n_select = 10
save_dir = "selected_heads_layers_plots"
os.makedirs(save_dir, exist_ok=True)
sns.set(style="darkgrid")

def compute_concentration_metric(values, indices, metric):
    w = np.array(values, dtype=float)
    if w.ndim != 3:
        return None
    idxs = np.array(indices, dtype=int)
    if idxs.ndim != 3:
        return None
    n_layers_, n_heads_, n_top = w.shape
    x = (idxs % grid_size + 0.5) / grid_size
    y = (idxs // grid_size + 0.5) / grid_size
    out = np.zeros((n_layers_, n_heads_))
    for l in range(n_layers_):
        for h in range(n_heads_):
            vals = w[l, h]
            probs = vals / (np.sum(vals) + eps)
            if metric == "entropy":
                out[l, h] = -np.sum(probs * np.log(probs + eps))
            elif metric == "gini":
                sorted_p = np.sort(probs)
                n = len(sorted_p)
                coef = 2 * np.arange(1, n + 1) - n - 1
                out[l, h] = abs(np.sum(coef * sorted_p) / (n * np.sum(sorted_p) + eps))
            else:
                out[l, h] = np.sum(probs)
    return out

def extract_metric_values(data_dict, cls_, metric):
    results = []
    entries = data_dict.get(cls_, {}).get("image", [])
    for e in entries:
        if not e.get("subtoken_results"):
            continue
        for sub in e["subtoken_results"][:n_subtokens]:
            topk_vals = np.array(sub["topk_values"], dtype=float)
            topk_inds = np.array(sub["topk_indices"], dtype=int)
            if topk_vals.ndim != 3 or topk_inds.ndim != 3:
                continue
            idx = int(sub["idx"])
            if metric == "attention":
                mean_vals = np.mean(topk_vals[..., :n_top_k], axis=-1)
                results.append((idx, mean_vals))
            else:
                conc = compute_concentration_metric(topk_vals[..., :n_top_k], topk_inds[..., :n_top_k], metric)
                results.append((idx, conc))
    return results

def aggregate_across_images(files, n_files, metric):
    tp_collect = []
    fp_collect = []
    for f in tqdm(files[:n_files]):
        try:
            with open(f, "rb") as handle:
                data_dict = pickle.load(handle)
        except Exception:
            continue
        tp_collect.extend(extract_metric_values(data_dict, "tp", metric))
        fp_collect.extend(extract_metric_values(data_dict, "fp", metric))
    return tp_collect, fp_collect
