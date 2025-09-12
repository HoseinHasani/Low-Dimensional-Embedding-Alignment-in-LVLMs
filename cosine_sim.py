import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

use_baseline = False  

vis = np.load("data/just_number_L25/visual_row_means_L25.npy")
txt = np.load("data/just_number_L25/text_fullline_rows_L25.npy")

if use_baseline:
    vis_baseline = np.load("data/just_number_L25/visual_row_means_L25_baseline.npy")
    vis = vis - vis_baseline

def clean_embeddings(x):
    for i in range(x.shape[1]):
        col = x[:, i]
        if np.isnan(col).any():
            mean_val = np.nanmean(col)
            if np.isnan(mean_val):
                mean_val = 0.0
            col = np.nan_to_num(col, nan=mean_val)
        x[:, i] = col
    return x



txt_all = np.vstack([clean_embeddings(txt[n]) for n in range(txt.shape[0])])
vis_all = np.vstack([clean_embeddings(vis[n]) for n in range(vis.shape[0])])

txt_mean, txt_std = txt_all.mean(axis=0), txt_all.std(axis=0)
vis_mean, vis_std = vis_all.mean(axis=0), vis_all.std(axis=0)

all_sims = []

for n in range(vis.shape[0]):
    txt_emb = clean_embeddings(txt[n])
    vis_emb = clean_embeddings(vis[n])


    txt_emb = (txt_emb - txt_mean) #/ (txt_std + 1e-8)
    vis_emb = (vis_emb - vis_mean) #/ (vis_std + 1e-8)

    txt_emb = normalize(txt_emb.T, norm="l2")
    vis_emb = normalize(vis_emb.T, norm="l2")

    sim = cosine_similarity(txt_emb, vis_emb)
    sim = np.where(np.isnan(sim), np.nanmean(sim), sim)
    
    if False:
        if sim.max() > 0:
            sim = sim / sim.max()

    all_sims.append(sim)

mean_sim = np.mean(all_sims, axis=0)

plt.figure(figsize=(6,5))
im = plt.imshow(mean_sim, cmap="coolwarm", vmin=mean_sim.min(), vmax=mean_sim.max())
plt.colorbar(im)
for i in range(4):
    for j in range(4):
        plt.text(j, i, f"{mean_sim[i, j]:.2f}", ha="center", va="center", color="black")
plt.xticks(range(4), [f"vis{j+1}" for j in range(4)])
plt.yticks(range(4), [f"txt{i+1}" for i in range(4)])
plt.title("Average Cosine Similarity (Z-score + L2 Normalized)" + ("\nBaseline Subtracted" if use_baseline else ""))
plt.show()
