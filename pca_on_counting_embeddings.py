import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.cm import get_cmap


data_path = "data/counting_pca3"
token_mode = "separator"          # "element", "separator", or "both"
mean_over_dataset = False   # True: average across experiments; False: concatenate all


def load_embeddings(layer_path, token_mod):
    data = {"element": [], "separator": []}
    for file in os.listdir(layer_path):
        if not file.endswith(".pt"):
            continue
        token_type = "element" if file.startswith("element") else "separator"
        if token_mod in ["both", token_type]:
            m = re.search(r"_(\d+)\.pt$", file)
            if not m:
                print(f"Skipping file with unexpected name: {file}")
                continue
            token_idx = int(m.group(1))
            tensor = torch.load(os.path.join(layer_path, file), map_location="cpu")
            tensor = tensor.detach().cpu().numpy().reshape(-1)
            data[token_type].append((token_idx, tensor))
    return data


def aggregate_layer_data(layer_dirs, token_mod, mean_over_dataset):
    all_data = {"element": {}, "separator": {}}

    for layer_dir in layer_dirs:
        layer_data = load_embeddings(layer_dir, token_mod)
        for token_type, files in layer_data.items():
            for token_idx, vec in files:
                if token_idx not in all_data[token_type]:
                    all_data[token_type][token_idx] = []
                all_data[token_type][token_idx].append(vec)

    combined_data = {"element": [], "separator": []}
    for token_type, token_dict in all_data.items():
        for idx, vecs in token_dict.items():
            if not vecs:
                continue
            if mean_over_dataset:
                combined = np.mean(np.stack(vecs), axis=0)
                combined_data[token_type].append((idx, combined))
            else:
                for v in vecs:
                    combined_data[token_type].append((idx, v))
    return combined_data


def apply_pca_and_plot(ax, data, token_mod, layer_name, cmap):
    all_vecs, labels, shapes = [], [], []

    if token_mod in ["element", "both"]:
        for idx, vec in data["element"]:
            all_vecs.append(vec)
            labels.append(idx)
            shapes.append("o")

    if token_mod in ["separator", "both"]:
        for idx, vec in data["separator"]:
            all_vecs.append(vec)
            labels.append(idx)
            shapes.append("*")

    if len(all_vecs) == 0:
        ax.set_title(f"{layer_name} (no data)")
        ax.axis("off")
        return

    X = np.stack(all_vecs)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    unique_tokens = sorted(set(labels))
    color_map = {tok: cmap(i / max(1, len(unique_tokens) - 1)) for i, tok in enumerate(unique_tokens)}

    for (x, y), tok, shape in zip(X_pca, labels, shapes):
        ax.scatter(x, y, marker=shape, color=color_map[tok],
                   edgecolor="black", s=80, alpha=0.85)

    ax.set_title(layer_name, fontsize=14, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")



experiments = [
    os.path.join(data_path, exp)
    for exp in os.listdir(data_path)
    if os.path.isdir(os.path.join(data_path, exp))
]

if not experiments:
    raise ValueError(f"No experiments found in {data_path}")

sample_exp = experiments[0]
layer_names = sorted(
    [d for d in os.listdir(sample_exp) if d.startswith("layer_")]
)

n_layers = len(layer_names)
nrows, ncols = 4, 7  # fixed 4x7 layout
fig, axes = plt.subplots(nrows, ncols, figsize=(28, 16))
axes = axes.flatten()

cmap = get_cmap("tab10")

for i, layer_name in enumerate(layer_names[: nrows * ncols]):
    layer_dirs = [
        os.path.join(exp, layer_name)
        for exp in experiments
        if os.path.exists(os.path.join(exp, layer_name))
    ]
    if not layer_dirs:
        axes[i].set_title(f"{layer_name} (missing)")
        axes[i].axis("off")
        continue

    data = aggregate_layer_data(layer_dirs, token_mode, mean_over_dataset)
    apply_pca_and_plot(axes[i], data, token_mode, layer_name, cmap)

for j in range(len(layer_names), len(axes)):
    axes[j].axis("off")

handles = [
    plt.Line2D([], [], color=cmap(i / 8), marker='o', linestyle='', label=f"Token {i:02d}")
    for i in range(9)
]
if token_mode == "both":
    handles += [
        plt.Line2D([], [], color="black", marker='o', linestyle='', label="Element"),
        plt.Line2D([], [], color="black", marker='*', linestyle='', label="Separator"),
    ]

fig.legend(handles=handles, loc="upper center", ncol=10,
           frameon=False, fontsize=14, bbox_to_anchor=(0.5, 1.03))
fig.suptitle(f"PCA Visualization of Layer Embeddings ({token_mode})",
             fontsize=26, weight="bold", y=0.97)

plt.subplots_adjust(top=0.90, bottom=0.05, left=0.04, right=0.98,
                    hspace=0.4, wspace=0.25)

out_name = f"layerwise_pca_{token_mode}_pe.png"
plt.savefig(out_name, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved PCA visualization as {out_name}")


