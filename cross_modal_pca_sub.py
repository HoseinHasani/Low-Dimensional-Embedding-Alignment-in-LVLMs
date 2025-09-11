import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

use_baseline = True  
baseline_mode = "post"
normalize_std = False
pca_dim = 3
clf_source = "txt"  

vis = np.load("data/just_number_L25/visual_row_means_L25.npy")
txt = np.load("data/just_number_L25/text_fullline_rows_L25.npy")


if use_baseline:
    vis_baseline = np.load("data/just_number_L25/visual_row_means_L25_baseline.npy")
    if baseline_mode == "pre":
        vis = vis - vis_baseline

def clean_array(x):
    x2 = x.copy()
    for d in range(x2.shape[1]):
        col = x2[:, d]
        if np.isnan(col).any():
            mean_val = np.nanmean(col)
            if np.isnan(mean_val):
                mean_val = 0.0
            col = np.nan_to_num(col, nan=mean_val)
        x2[:, d] = col
    return x2

txt_clean = []
vis_clean = []
for n in range(txt.shape[0]):
    txt_clean.append(clean_array(txt[n].T))
    vis_clean.append(clean_array(vis[n].T))
txt_clean = np.vstack(txt_clean) 
vis_clean = np.vstack(vis_clean) 

txt_mean, txt_std = txt_clean.mean(axis=0), txt_clean.std(axis=0)
vis_mean, vis_std = vis_clean.mean(axis=0), vis_clean.std(axis=0)


txt_clean = (txt_clean - txt_mean)
vis_clean = (vis_clean - vis_mean)

if normalize_std:
    txt_clean = txt_clean / (txt_std + 1e-8)
    vis_clean = vis_clean / (vis_std + 1e-8)

pca = PCA(n_components=pca_dim)
pca.fit(txt_clean)

colors = ["red", "blue", "green", "purple"]

txt_proj = []
vis_proj = []
vis_baseline_proj = [] if (use_baseline and baseline_mode == "post") else None

for n in range(vis.shape[0]):
    txt_emb = (clean_array(txt[n].T) - txt_mean) 
    vis_emb = (clean_array(vis[n].T) - vis_mean)

    if normalize_std:
        txt_emb = txt_emb / (txt_std + 1e-8)
        vis_emb = vis_emb / (vis_std + 1e-8)

    txt_emb_pca = pca.transform(txt_emb)
    vis_emb_pca = pca.transform(vis_emb)

    txt_proj.append(txt_emb_pca)
    vis_proj.append(vis_emb_pca)

    if vis_baseline_proj is not None:
        vis_base_emb = (clean_array(vis_baseline[n].T) - vis_mean)
        if normalize_std:
            vis_base_emb = vis_base_emb / (vis_std + 1e-8)
        vis_baseline_proj.append(pca.transform(vis_base_emb))

txt_proj = np.array(txt_proj)
vis_proj = np.array(vis_proj)

if vis_baseline_proj is not None:
    vis_baseline_proj = np.array(vis_baseline_proj)
    vis_proj = vis_proj - vis_baseline_proj




all_sims = []
for n in range(vis.shape[0]):
    txt_emb_norm = normalize(txt_proj[n], norm="l2")
    vis_emb_norm = normalize(vis_proj[n], norm="l2")

    sim = cosine_similarity(txt_emb_norm, vis_emb_norm)
    sim = np.where(np.isnan(sim), np.nanmean(sim), sim)
    all_sims.append(sim)

mean_sim = np.mean(all_sims, axis=0)


txt_features = np.mean(txt_proj, axis=1)
vis_features = np.mean(vis_proj, axis=1)


num_samples = txt_proj.shape[0]
labels = np.tile(np.arange(4), num_samples // 4)

if clf_source == "txt":
    clf = LogisticRegression(max_iter=1000)
    clf.fit(txt_features, labels)
    preds = clf.predict(vis_features)
    acc = accuracy_score(labels, preds)
    print(f"Train on TEXT, test on VISION → Accuracy: {acc:.3f}")

elif clf_source == "vis":
    clf = LogisticRegression(max_iter=1000)
    clf.fit(vis_features, labels)
    preds = clf.predict(txt_features)
    acc = accuracy_score(labels, preds)
    print(f"Train on VISION, test on TEXT → Accuracy: {acc:.3f}")


plt.figure(figsize=(8,7))
for i in range(4):
    xs = [txt_proj[n][i,0] for n in range(len(txt_proj))]
    ys = [txt_proj[n][i,1] for n in range(len(txt_proj))]
    plt.scatter(xs, ys, color=colors[i], alpha=0.6, label=f"txt{i+1}")
plt.title("Text PCA Projection (all samples)")
plt.legend()
plt.show()

plt.figure(figsize=(8,7))
for i in range(4):
    xs = [vis_proj[n][i,0] for n in range(len(vis_proj))]
    ys = [vis_proj[n][i,1] for n in range(len(vis_proj))]
    plt.scatter(xs, ys, color=colors[i], alpha=0.6, label=f"vis{i+1}")
plt.title("Vision PCA Projection (all samples)")
plt.legend()
plt.show()

plt.figure(figsize=(8,7))
im = plt.imshow(mean_sim, cmap="coolwarm", vmin=mean_sim.min(), vmax=mean_sim.max())
plt.colorbar(im)
for i in range(4):
    for j in range(4):
        plt.text(j, i, f"{mean_sim[i, j]:.2f}", ha="center", va="center", color="black")

plt.xticks(range(4), [f"vis{j+1}" for j in range(4)])
plt.yticks(range(4), [f"txt{i+1}" for i in range(4)])
plt.title(f"Average Cosine Similarity (PCA dim={pca_dim}, "
          + ("\nBaseline Subtracted" if use_baseline else ""))
plt.show()
