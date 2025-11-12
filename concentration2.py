import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import seaborn as sns

data_dir = "data/all layers all attention tp fp"
files = glob(os.path.join(data_dir, "attentions_*.pkl"))
n_layers, n_heads = 32, 32          
n_img_side = 24                     
top_k = 5                            
n_files = len(files)
sns.set(style="whitegrid")

def extract_topk_positions(data_dict, cls_, top_k):
    """Return list of (token_idx, topk_pos[l,h,top_k]) for one class (tp/fp/other)."""
    results = []
    entries = data_dict.get(cls_, {}).get("image", [])
    for e in entries:
        if not e.get("subtoken_results"):
            continue
        for sub in e["subtoken_results"]:
            topk_inds = np.array(sub["topk_indices"], dtype=int)
            if topk_inds.ndim != 3:
                continue
            k = min(topk_inds.shape[-1], top_k)
            topk_inds = topk_inds[..., :k]  # shape (n_layers, n_heads, top_k)
            token_idx = int(sub["idx"])
            results.append((token_idx, topk_inds))
    return results

def aggregate_topk_positions(files, n_files, top_k):
    tp, fp, oth = [], [], []
    for f in tqdm(files[:n_files]):
        try:
            with open(f, "rb") as handle:
                data_dict = pickle.load(handle)
        except Exception:
            continue
        tp.extend(extract_topk_positions(data_dict, "tp", top_k))
        fp.extend(extract_topk_positions(data_dict, "fp", top_k))
        oth.extend(extract_topk_positions(data_dict, "other", top_k))
    return tp, fp, oth

def compute_variance_by_layer(attn_data, n_layers, n_heads, n_img_side=24):

    layer_token_positions = {l: [] for l in range(n_layers)}
    layer_variances = {l: [] for l in range(n_layers)}

    for token_idx, topk_inds in attn_data:
        rows = topk_inds // n_img_side    # shape (n_layers, n_heads, top_k)
        cols = topk_inds % n_img_side

        for l in range(n_layers):
            # Flatten across heads and top_k
            all_rows = rows[l].flatten()
            all_cols = cols[l].flatten()

            row_var = np.var(all_rows)
            col_var = np.var(all_cols)
            total_var = row_var + col_var   # total 2D spatial dispersion

            layer_variances[l].append(total_var)
            layer_token_positions[l].append(token_idx)

    return layer_token_positions, layer_variances

def plot_combined_variance(tp_pos, tp_var, fp_pos, fp_var, oth_pos, oth_var, save_dir, top_k):
    os.makedirs(save_dir, exist_ok=True)
    n_layers = len(tp_var)

    for l in range(n_layers):
        plt.figure(figsize=(9, 5))

        def layer_curve(layer_pos, layer_var):
            if not layer_var[l]:
                return None, None, None
            x = np.array(layer_pos[l])
            y = np.array(layer_var[l])
            uniq_x = sorted(set(x))
            mean_y = np.array([np.mean(y[x == pos]) for pos in uniq_x])
            std_y = np.array([np.std(y[x == pos]) for pos in uniq_x])
            return np.array(uniq_x), mean_y, std_y

        tp_x, tp_y, tp_std = layer_curve(tp_pos, tp_var)
        fp_x, fp_y, fp_std = layer_curve(fp_pos, fp_var)
        oth_x, oth_y, oth_std = layer_curve(oth_pos, oth_var)

        # Plot with shaded error bars
        def plot_with_std(x, y, std, color, label):
            if x is None: return
            plt.plot(x, y, color=color, lw=1.8, label=label)
            plt.fill_between(x, y - std, y + std, color=color, alpha=0.2)

        plot_with_std(tp_x, tp_y, tp_std, "tab:green", "TP")
        plot_with_std(fp_x, fp_y, fp_std, "tab:red", "FP")
        plot_with_std(oth_x, oth_y, oth_std, "tab:gray", "Other")
        
        plt.xlim(4, 150)

        plt.xlabel("Token Position in Generated Text", fontsize=13)
        plt.ylabel("Variance of Attended Image Positions", fontsize=13)
        plt.title(f"Layer {l+1} – Attention Concentration (top-{top_k})", fontsize=15)
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"layer_{l+1}_top{top_k}_tp_fp_other.png"), dpi=150)
        plt.close()

print(f"Extracting top-{top_k} attention positions from all files...")
tp_data, fp_data, oth_data = aggregate_topk_positions(files, n_files, top_k)

print("Computing variance across heads and top_k positions per layer...")
tp_pos, tp_var = compute_variance_by_layer(tp_data, n_layers, n_heads)
fp_pos, fp_var = compute_variance_by_layer(fp_data, n_layers, n_heads)
oth_pos, oth_var = compute_variance_by_layer(oth_data, n_layers, n_heads)

print("Plotting combined TP/FP/Other variance curves with STD bands...")
save_dir = f"variance_plots_combined_top{top_k}_std"
plot_combined_variance(tp_pos, tp_var, fp_pos, fp_var, oth_pos, oth_var, save_dir, top_k)

print(f"Done! Variance plots with STD saved in '{save_dir}/'")

