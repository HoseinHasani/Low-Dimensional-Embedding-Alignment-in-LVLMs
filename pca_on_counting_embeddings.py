import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.cm import get_cmap

data_path = "data/counting pca"
token_mod = "both"  # "element", "separator", or "both"
mean_over_dataset = False



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
        ax.set_title(layer_name + " (no data)")
        return
    X = np.stack(all_vecs)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    colors = [cmap(l / 8) for l in labels]
    for (x, y), color, shape in zip(X_pca, colors, shapes):
        ax.scatter(x, y, marker=shape, color=color, edgecolor="black", s=40, alpha=0.8)
    ax.set_title(layer_name, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    
def aggregate_layer_data(layer_dirs, token_mod, mean_over_dataset):
    all_data = {"element": {}, "separator": {}}
    for token_type in all_data.keys():
        for i in range(9):
            all_data[token_type][i] = []
    for layer_dir in layer_dirs:
        layer_data = load_embeddings(layer_dir, token_mod)
        for token_type, files in layer_data.items():
            for fname, vec in files:
                token_idx = int(fname.split("_")[1].split(".")[0])
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

def load_embeddings(layer_path, token_mod):
    data = {"element": [], "separator": []}
    for file in os.listdir(layer_path):
        if file.endswith(".pt"):
            token_type = "element" if file.startswith("element") else "separator"
            if token_mod in ["both", token_type]:
                tensor = torch.load(os.path.join(layer_path, file), map_location="cpu")
                tensor = tensor.detach().cpu().numpy().reshape(-1)
                data[token_type].append((file, tensor))
    return data


experiments = [os.path.join(data_path, exp) for exp in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, exp))]
sample_exp = experiments[0]
layer_names = sorted([d for d in os.listdir(sample_exp) if d.startswith("layer_")])
fig, axes = plt.subplots(4, 7, figsize=(21, 12))
axes = axes.flatten()
cmap = get_cmap("tab10")

for i, layer_name in enumerate(layer_names[:28]):
    layer_dirs = [os.path.join(exp, layer_name) for exp in experiments if os.path.exists(os.path.join(exp, layer_name))]
    data = aggregate_layer_data(layer_dirs, token_mod, mean_over_dataset)
    apply_pca_and_plot(axes[i], data, token_mod, layer_name, cmap)

for j in range(len(layer_names), len(axes)):
    axes[j].axis("off")

handles = []
for i in range(9):
    handles.append(plt.Line2D([], [], color=cmap(i / 8), marker='o', linestyle='', label=f"Token {i:02d}"))
if token_mod == "both":
    handles += [
        plt.Line2D([], [], color="black", marker='o', linestyle='', label="Element"),
        plt.Line2D([], [], color="black", marker='*', linestyle='', label="Separator"),
    ]

fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, fontsize=9)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.suptitle("PCA Visualization of Layer Embeddings", fontsize=14, weight="bold")
plt.savefig(f"layerwise_pca_{token_mod}.png", dpi=160, bbox_inches="tight")
plt.close()
