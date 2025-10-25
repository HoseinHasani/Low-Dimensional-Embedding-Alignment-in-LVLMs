import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.cm import get_cmap

data_path = "data/counting_pca2"
mean_over_dataset = False
label_mode = "number"   # "fruit" or "number"
nrows, ncols = 4, 7

def load_layer_embeddings(experiments, layer_name, mean_over_dataset):
    layer_vecs = []
    for exp in experiments:
        layer_file = os.path.join(exp, layer_name + ".pt")
        if not os.path.exists(layer_file):
            continue
        tensor = torch.load(layer_file, map_location="cpu")
        if tensor.dtype in (torch.bfloat16, torch.float16):
            tensor = tensor.to(torch.float32)
        tensor = tensor.detach().cpu().numpy().reshape(-1)
        layer_vecs.append(tensor)
    if not layer_vecs:
        return None
    if mean_over_dataset:
        return np.mean(np.stack(layer_vecs), axis=0, keepdims=True)
    else:
        return np.stack(layer_vecs)

def apply_pca_and_plot(ax, X, labels, layer_name, cmap):
    if X is None or len(X) < 2:
        ax.set_title(f"{layer_name} (no data)")
        ax.axis("off")
        return
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    unique_labels = sorted(set(labels))
    color_map = {label: cmap(i / max(1, len(unique_labels) - 1))
                 for i, label in enumerate(unique_labels)}
    for (x, y), lbl in zip(X_pca, labels):
        ax.scatter(x, y, color=color_map[lbl], edgecolor="black", s=80, alpha=0.9)
    ax.set_title(layer_name, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")

experiments = [
    os.path.join(data_path, d)
    for d in os.listdir(data_path)
    if os.path.isdir(os.path.join(data_path, d))
]

if not experiments:
    raise ValueError(f"No experiments found in {data_path}")

sample_exp = experiments[0]
layer_files = sorted(
    [f for f in os.listdir(sample_exp) if f.startswith("layer_") and f.endswith(".pt")]
)
layer_names = [f[:-3] for f in layer_files]

def get_label(exp_path):
    base = os.path.basename(exp_path)
    parts = base.split("_")
    if label_mode == "fruit":
        return parts[0]
    elif label_mode == "number":
        return parts[-1]
    else:
        raise ValueError(f"Invalid label_mode: {label_mode}")

labels_all = [get_label(e) for e in experiments]

fig, axes = plt.subplots(nrows, ncols, figsize=(28, 16))
axes = axes.flatten()
cmap = get_cmap("tab10")

for i, layer_name in enumerate(layer_names[: nrows * ncols]):
    X = load_layer_embeddings(experiments, layer_name, mean_over_dataset)
    if X is not None:
        apply_pca_and_plot(axes[i], X, labels_all, layer_name, cmap)
    else:
        axes[i].set_title(f"{layer_name} (missing)")
        axes[i].axis("off")

for j in range(len(layer_names), len(axes)):
    axes[j].axis("off")

unique_labels = sorted(set(labels_all))
handles = [
    plt.Line2D([], [], color=cmap(i / max(1, len(unique_labels) - 1)),
               marker='o', linestyle='', label=str(lbl))
    for i, lbl in enumerate(unique_labels)
]
fig.legend(handles=handles, loc="upper center", ncol=10,
           frameon=False, fontsize=14, bbox_to_anchor=(0.5, 1.03))

fig.suptitle(f"PCA of Generated Token Embeddings (per Layer) — labeled by {label_mode}",
             fontsize=26, weight="bold", y=0.97)

plt.subplots_adjust(top=0.90, bottom=0.05, left=0.04, right=0.98,
                    hspace=0.4, wspace=0.25)

out_name = f"layerwise_pca_generated_responses_{label_mode}.png"
plt.savefig(out_name, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved PCA visualization as {out_name}")
