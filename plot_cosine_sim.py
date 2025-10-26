import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt

data_path = "data/counting_pca3"
token_mode = "element"    # "element" or "separator"
nrows, ncols = 4, 7       # grid for 28 layers
out_name = f"layerwise_cosine_diff_{token_mode}.png"


def load_token_embeddings(layer_path, token_type):
    """Load all token embeddings of a given type (element/separator) from one layer."""
    data = {}
    for file in os.listdir(layer_path):
        if not file.endswith(".pt") or not file.startswith(token_type):
            continue
        m = re.search(r"_(\d+)\.pt$", file)
        if not m:
            continue
        token_idx = int(m.group(1))
        vec = torch.load(os.path.join(layer_path, file), map_location="cpu")
        data[token_idx] = vec.detach().cpu().numpy().flatten()
    return data


def compute_diff_vectors(token_data):
    """Compute normalized difference vectors grouped by gap size."""
    diffs = {gap: [] for gap in range(1, 9)}  # possible gaps: 1..8
    for a in range(9):
        for b in range(a):
            gap = a - b
            if gap > 8:
                continue
            diff = token_data[a] - token_data[b]
            norm = np.linalg.norm(diff)
            if norm > 0:
                diff /= norm
            diffs[gap].append(diff)
    return diffs


def layer_cosine_similarity(layer_dirs, token_mode):
    experiments_diffs = []
    for layer_dir in layer_dirs:
        tokens = load_token_embeddings(layer_dir, token_mode)
        if len(tokens) < 9:
            continue
        diffs = compute_diff_vectors(tokens)
        experiments_diffs.append(diffs)

    n_exps = len(experiments_diffs)
    if n_exps < 2:
        return np.zeros((8, 8))

    sim_mats = []

    # Compare across pairs of experiments
    for i in range(n_exps):
        for j in range(i + 1, n_exps):
            diffs_i = experiments_diffs[i]
            diffs_j = experiments_diffs[j]
            mat = np.zeros((8, 8))
            for g1 in range(1, 9):
                for g2 in range(1, 9):
                    if not diffs_i[g1] or not diffs_j[g2]:
                        mat[g1 - 1, g2 - 1] = np.nan
                        continue
                    v1 = np.stack(diffs_i[g1])
                    v2 = np.stack(diffs_j[g2])
                    cos = v1 @ v2.T
                    cos_mean = np.mean(cos)
                    mat[g1 - 1, g2 - 1] = cos_mean
            sim_mats.append(mat)

    sim_mats = np.stack(sim_mats)
    mean_mat = np.nanmean(sim_mats, axis=0)
    return mean_mat


experiments = [
    os.path.join(data_path, exp)
    for exp in os.listdir(data_path)
    if os.path.isdir(os.path.join(data_path, exp))
]

if not experiments:
    raise ValueError(f"No experiments found in {data_path}")

sample_exp = experiments[0]
layer_names = sorted([d for d in os.listdir(sample_exp) if d.startswith("layer_")])
n_layers = len(layer_names)

fig, axes = plt.subplots(nrows, ncols, figsize=(28, 16))
axes = axes.flatten()

print("Computing cosine similarity matrices...")
layer_mats = []
for layer_name in layer_names:
    layer_dirs = [os.path.join(exp, layer_name) for exp in experiments if os.path.exists(os.path.join(exp, layer_name))]
    if not layer_dirs:
        layer_mats.append(None)
        continue
    mat = layer_cosine_similarity(layer_dirs, token_mode)
    layer_mats.append(mat)

all_vals = np.concatenate([m.flatten() for m in layer_mats if m is not None])
vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
print(f"Color range: {vmin:.3f} to {vmax:.3f}")

for i, (layer_name, mat) in enumerate(zip(layer_names[:28], layer_mats[:28])):
    ax = axes[i]
    if mat is None:
        ax.axis("off")
        continue
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="coolwarm", origin="lower")
    ax.set_title(layer_name, fontsize=12)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels(range(1, 9))
    ax.set_yticklabels(range(1, 9))
    if i % ncols == 0:
        ax.set_ylabel("Gap i")
    if i // ncols == nrows - 1:
        ax.set_xlabel("Gap j")

for j in range(len(layer_names), len(axes)):
    axes[j].axis("off")

cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
fig.colorbar(im, cax=cbar_ax, label="Cosine Similarity")

fig.suptitle(f"Layerwise Cosine Similarity of ΔVectors ({token_mode})", fontsize=24, weight="bold")
plt.subplots_adjust(top=0.90, bottom=0.05, left=0.04, right=0.92, hspace=0.4, wspace=0.3)

plt.savefig(out_name, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved cosine similarity figure as {out_name}")
