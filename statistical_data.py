import os
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
from scipy.stats import ttest_ind


data_dir = "data/all layers all attention tp fp"
files = glob(os.path.join(data_dir, "attentions_*.pkl"))

n_files = 3900
n_layers, n_heads = 32, 32
n_top_k = 20
n_subtokens = 1
eps = 1e-130

position_min = 5
position_max = 160
position_margin = 3

save_root = "stats_summary"
os.makedirs(save_root, exist_ok=True)


def compute_entropy(vals):
    if vals.ndim != 3:
        return None
    probs = vals / (np.sum(vals, axis=-1, keepdims=True) + eps)
    return -np.sum(probs * np.log(probs + eps), axis=-1)


def compute_gini(vals):
    if vals.ndim != 3:
        return None
    probs = vals / (np.sum(vals, axis=-1, keepdims=True) + eps)
    sorted_p = np.sort(probs, axis=-1)
    n = sorted_p.shape[-1]
    coef = 2 * np.arange(1, n+1) - n - 1
    gini = np.sum(coef * sorted_p, axis=-1) / (n * np.sum(sorted_p, axis=-1) + eps)
    return np.abs(gini)


def extract_attention_values(data_dict, cls_):
    results = []
    entries = data_dict.get(cls_, {}).get("image", [])
    for e in entries:
        if not e.get("subtoken_results"):
            continue
        for sub in e["subtoken_results"][:n_subtokens]:
            vals = np.array(sub["topk_values"], dtype=float)
            if vals.ndim != 3:
                continue

            idx = int(sub["idx"])
            mean_vals = np.mean(vals[..., :n_top_k], axis=-1)
            ent_vals  = compute_entropy(vals[..., :n_top_k])
            gin_vals  = compute_gini(vals[..., :n_top_k])

            results.append((idx, mean_vals, ent_vals, gin_vals))
    return results


def aggregate_across_images(files, n_files):
    tp_collect, fp_collect, oth_collect = [], [], []
    for f in tqdm(files[:n_files], desc="Loading files"):
        try:
            with open(f, "rb") as handle:
                data_dict = pickle.load(handle)
        except Exception:
            continue

        tp_collect.extend(extract_attention_values(data_dict, "tp"))
        fp_collect.extend(extract_attention_values(data_dict, "fp"))
        oth_collect.extend(extract_attention_values(data_dict, "other"))
    return tp_collect, fp_collect, oth_collect


# -----------------------
def aggregate_by_position(data, metric_id):
    layer_head_map = {L: {H: {} for H in range(n_heads)} for L in range(n_layers)}
    for idx, att, ent, gin in data:
        metric = [att, ent, gin][metric_id]
        if metric is None:
            continue

        for L in range(n_layers):
            for H in range(n_heads):
                layer_head_map[L][H].setdefault(idx, []).append(float(metric[L, H]))

    return layer_head_map

def compute_pvalue_map(tp_map, fp_map):
    pos_to_logp = {}

    for pos in range(position_min, position_max + 1):
        fp_vals, tp_vals = [], []

        # gather values in the window
        for p in range(pos - position_margin, pos + position_margin + 1):
            fp_vals.extend(fp_map.get(p, []))
            tp_vals.extend(tp_map.get(p, []))

        if len(fp_vals) > 1 and len(tp_vals) > 1:
            _, p_val = ttest_ind(fp_vals, tp_vals, equal_var=False)
            p_val = max(p_val, eps)
            pos_to_logp[pos] = np.log10(p_val)
        else:
            pos_to_logp[pos] = np.log10(1.0)  # no evidence

    return pos_to_logp



tp_data, fp_data, oth_data = aggregate_across_images(files, n_files)

metric_names = ["attention", "entropy", "gini"]

tp_maps = [aggregate_by_position(tp_data, i) for i in range(3)]
fp_maps = [aggregate_by_position(fp_data, i) for i in range(3)]

for midx, metric in enumerate(metric_names):
    metric_dir = os.path.join(save_root, metric)
    os.makedirs(metric_dir, exist_ok=True)

    print(f"\n=== Processing metric: {metric} ===")

    for L in tqdm(range(n_layers), desc=f"{metric} layers"):
        for H in range(n_heads):

            tp_map = tp_maps[midx][L][H]
            fp_map = fp_maps[midx][L][H]

            stats_dict = compute_pvalue_map(tp_map, fp_map)

            save_path = os.path.join(metric_dir, f"L_{L}_H_{H}.npy")
            np.save(save_path, stats_dict, allow_pickle=True)

print("\nAll statistical maps saved successfully!")
