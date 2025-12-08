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
n_layers, n_heads = 32, 32           # as in your model
n_img_side = 24                      # 24x24 = 576 visual tokens
top_k = 2                            # <-- choose how many top positions to include
n_files = len(files)
sns.set(style="whitegrid")

# ---------------- STEP 1: Extract top-k attended image positions ----------------
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

# ---------------- STEP 2: Aggregate across images ----------------
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

# ---------------- STEP 3: Compute variance for each token position ----------------
def compute_variance_for_token(attn_data, n_layers, n_heads, n_img_side=24, top_k=5):
    """
    attn_data: list of (token_idx, topk_indices[l,h,k])
    Returns: variance across heads, layers, and top_k for all tokens
    """
    token_variances = {}

    for token_idx, topk_inds in attn_data:
        rows = topk_inds // n_img_side    # shape (n_layers, n_heads, top_k)
        cols = topk_inds % n_img_side

        # Flatten across layers, heads, and top_k
        all_rows = rows.flatten()
        all_cols = cols.flatten()

        # Variance across all heads, top_k positions, and layers for this token
        row_var = np.var(all_rows)
        col_var = np.var(all_cols)
        total_var = row_var + col_var  # total 2D spatial variance

        if token_idx not in token_variances:
            token_variances[token_idx] = []
        token_variances[token_idx].append(total_var)

    return token_variances

# ---------------- STEP 4: Plot Variances ----------------
def plot_variance_over_tokens(tp_variances, fp_variances, oth_variances, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Prepare the variance data for plotting
    token_ids = sorted(set(tp_variances.keys()) | set(fp_variances.keys()) | set(oth_variances.keys()))
    
    tp_values = [np.mean(tp_variances.get(token, [0])) for token in token_ids]
    fp_values = [np.mean(fp_variances.get(token, [0])) for token in token_ids]
    oth_values = [np.mean(oth_variances.get(token, [0])) for token in token_ids]

    # Compute standard deviation (error bars)
    tp_std = [np.std(tp_variances.get(token, [0])) for token in token_ids]
    fp_std = [np.std(fp_variances.get(token, [0])) for token in token_ids]
    oth_std = [np.std(oth_variances.get(token, [0])) for token in token_ids]

    # Plotting the variances
    plt.figure(figsize=(10, 6))
    plt.errorbar(token_ids, tp_values, yerr=tp_std, label="tp", fmt='-o', color='b', capsize=5)
    plt.errorbar(token_ids, fp_values, yerr=fp_std, label="fp", fmt='-o', color='r', capsize=5)
    plt.errorbar(token_ids, oth_values, yerr=oth_std, label="other", fmt='-o', color='g', capsize=5)

    plt.xlim(4, 150)
    
    # Formatting plot
    plt.title(f"Variance of Attention Concentration Over Token Positions", fontsize=15)
    plt.xlabel("Token Position", fontsize=12)
    plt.ylabel("Variance of Attended Positions", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"variance_over_tokens.png"), dpi=150)
    plt.close()

# ---------------- STEP 5: Run everything ----------------
print(f"Extracting top-{top_k} attention positions from all files...")
tp_data, fp_data, oth_data = aggregate_topk_positions(files, n_files, top_k)

print("Computing variance for each token across layers, heads, and top_k positions...")
tp_variances = compute_variance_for_token(tp_data, n_layers, n_heads)
fp_variances = compute_variance_for_token(fp_data, n_layers, n_heads)
oth_variances = compute_variance_for_token(oth_data, n_layers, n_heads)

print("Plotting variance over token positions...")
save_dir = f"variance_plots_tokens_top{top_k}"
plot_variance_over_tokens(tp_variances, fp_variances, oth_variances, save_dir)

print(f"Done! Variance plot saved in '{save_dir}/'")

