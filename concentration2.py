import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import seaborn as sns

# ---------------- CONFIG ----------------
data_dir = "data/all layers all attention tp fp"
files = glob(os.path.join(data_dir, "attentions_*.pkl"))
n_layers, n_heads = 32, 32           # as in your topk arrays
n_img_side = 24                      # 24x24 = 576 visual tokens
n_files = len(files)
sns.set(style="whitegrid")

# ---------------- STEP 1: Extract top-1 attended image positions ----------------
def extract_max_positions(data_dict, cls_):
    """Return list of (token_idx, max_pos[l,h]) for one class (tp/fp/other)."""
    results = []
    entries = data_dict.get(cls_, {}).get("image", [])
    for e in entries:
        if not e.get("subtoken_results"):
            continue
        for sub in e["subtoken_results"]:
            topk_inds = np.array(sub["topk_indices"], dtype=int)
            if topk_inds.ndim != 3:
                continue
            max_inds = topk_inds[..., 0]  # shape (n_layers, n_heads)
            token_idx = int(sub["idx"])
            results.append((token_idx, max_inds))
    return results

# ---------------- STEP 2: Aggregate across images ----------------
def aggregate_max_positions(files, n_files):
    tp, fp, oth = [], [], []
    for f in tqdm(files[:n_files]):
        try:
            with open(f, "rb") as handle:
                data_dict = pickle.load(handle)
        except Exception:
            continue
        tp.extend(extract_max_positions(data_dict, "tp"))
        fp.extend(extract_max_positions(data_dict, "fp"))
        oth.extend(extract_max_positions(data_dict, "other"))
    return tp, fp, oth

# ---------------- STEP 3: Compute variance of 2D attention locations ----------------
def compute_variance_by_layer(attn_data, n_layers, n_heads, n_img_side=24):
    """
    attn_data: list of (token_idx, max_indices[l,h])
    Returns dict: layer -> (token_positions, variance_values)
    """
    layer_token_positions = {l: [] for l in range(n_layers)}
    layer_variances = {l: [] for l in range(n_layers)}

    for token_idx, max_inds in attn_data:
        rows = max_inds // n_img_side
        cols = max_inds % n_img_side

        for l in range(n_layers):
            row_var = np.var(rows[l, :])
            col_var = np.var(cols[l, :])
            total_var = row_var + col_var  # total spatial spread
            layer_variances[l].append(total_var)
            layer_token_positions[l].append(token_idx)

    return layer_token_positions, layer_variances

# ---------------- STEP 4: Plot TP/FP/Other together ----------------
def plot_combined_variance(tp_pos, tp_var, fp_pos, fp_var, oth_pos, oth_var, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    n_layers = len(tp_var)

    for l in range(n_layers):
        plt.figure(figsize=(9, 5))

        def layer_curve(layer_pos, layer_var):
            if not layer_var[l]:
                return None, None
            x = np.array(layer_pos[l])
            y = np.array(layer_var[l])
            uniq_x = sorted(set(x))
            mean_y = [np.mean(y[x == pos]) for pos in uniq_x]
            return np.array(uniq_x), np.array(mean_y)

        tp_x, tp_y = layer_curve(tp_pos, tp_var)
        fp_x, fp_y = layer_curve(fp_pos, fp_var)
        oth_x, oth_y = layer_curve(oth_pos, oth_var)

        if tp_x is not None:
            plt.plot(tp_x, tp_y, color="tab:green", label="TP", lw=1.8)
        if fp_x is not None:
            plt.plot(fp_x, fp_y, color="tab:red", label="FP", lw=1.8)
        if oth_x is not None:
            plt.plot(oth_x, oth_y, color="tab:gray", label="Other", lw=1.5)
            
        plt.xlim(4, 150)

        plt.xlabel("Token Position in Generated Text", fontsize=13)
        plt.ylabel("Variance of Attended Image Positions", fontsize=13)
        plt.title(f"Layer {l+1} – Attention Concentration", fontsize=15)
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"layer_{l+1}_tp_fp_other.png"), dpi=150)
        plt.close()

# ---------------- STEP 5: Run everything ----------------
print("Extracting top-1 attention positions from all files...")
tp_data, fp_data, oth_data = aggregate_max_positions(files, n_files)

print("Computing variance across heads per layer...")
tp_pos, tp_var = compute_variance_by_layer(tp_data, n_layers, n_heads)
fp_pos, fp_var = compute_variance_by_layer(fp_data, n_layers, n_heads)
oth_pos, oth_var = compute_variance_by_layer(oth_data, n_layers, n_heads)

print("Plotting combined TP/FP/Other variance curves...")
plot_combined_variance(tp_pos, tp_var, fp_pos, fp_var, oth_pos, oth_var, "variance_plots_combined")

print("Done! Combined variance plots saved in 'variance_plots_combined/'")

