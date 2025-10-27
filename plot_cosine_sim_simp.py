import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt

data_path = "data/counting_pca"
token_mode = "element"   # "element" or "separator"
nrows, ncols = 4, 7
out_name = f"layerwise_cosine_with_ref_{token_mode}.png"


def load_token_embeddings(layer_path, token_type):
    """Load all embeddings for a given token type from one layer directory."""
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


def compute_reference_diffs(token_data):
    """Compute normalized difference vectors relative to token_00."""
    diffs = []
    ref = token_data.get(0)
    if ref is None:
        return None
    for k in range(1, 9):
        if k not in token_data:
            return None
        diff = token_data[k] - ref
        norm = np.linalg.norm(diff)
        if norm > 0:
            diff /= norm
        diffs.append(diff)
    return np.stack(diffs)  # shape (8, dim)


def layer_cosine_similarity(layer_dirs, token_mode):
    """Compute average cosine similarity matrix between experiments for one layer."""
    exp_diffs = []
    for layer_dir in layer_dirs:
        tokens = load_token_embeddings(layer_dir, token_mode)
        diffs = compute_reference_diffs(tokens)
        if diffs is not None:
            exp_diffs.append(diffs)

    n_exps = len(exp_diffs)
    if n_exps < 2:
        return np.zeros((8, 8))

    sim_mats = []
    for i in range(n_exps):
        for j in range(i + 1, n_exps):
            v1 = exp_diffs[i]  # (8, dim)
            v2 = exp_diffs[j]  # (8, dim)
            sim = v1 @ v2.T    # (8, 8) cosine similarities
            sim_mats.append(sim)

    sim_mats = np.stack(sim_mats)
    return np.mean(sim_mats, axis=0)


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

print("Computing cosine similarity matrices (relative to token_00)...")
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

# Plot
for i, (layer_name, mat) in enumerate(zip(layer_names, layer_mats)):
    ax = axes[i]
    if mat is None:
        ax.axis("off")
        continue
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="coolwarm", origin="lower")
    ax.set_title(layer_name, fontsize=12)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels([f"{k:02d}" for k in range(1, 9)])
    ax.set_yticklabels([f"{k:02d}" for k in range(1, 9)])
    if i % ncols == 0:
        ax.set_ylabel("Exp A diffs")
    if i // ncols == nrows - 1:
        ax.set_xlabel("Exp B diffs")

for j in range(len(layer_names), len(axes)):
    axes[j].axis("off")

cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
fig.colorbar(im, cax=cbar_ax, label="Cosine Similarity")

fig.suptitle(f"Layerwise Cosine Similarity of Δ(token_k - token_00) ({token_mode})",
             fontsize=24, weight="bold")
plt.subplots_adjust(top=0.90, bottom=0.05, left=0.04, right=0.92, hspace=0.4, wspace=0.3)

plt.savefig(out_name, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved cosine similarity figure as {out_name}")
