import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

txt_embeddings = [np.load(f"data/layer25_just_number/trial-20_0_L25_txt_digit_row{i}.npy") for i in range(1,5)]
vis_embeddings = [np.load(f"data/layer25_just_number/trial-20_0_L25_vis_row{i}.npy") for i in range(1,5)]

txt_embeddings = normalize(np.vstack(txt_embeddings), norm="l2")
vis_embeddings = normalize(np.vstack(vis_embeddings), norm="l2")

sim_matrix = cosine_similarity(txt_embeddings, vis_embeddings)

plt.figure(figsize=(6,5))
im = plt.imshow(sim_matrix, cmap="coolwarm", vmin=sim_matrix.min(), vmax=sim_matrix.max())
plt.colorbar(im)
for i in range(4):
    for j in range(4):
        plt.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center", color="black")
plt.xticks(range(4), [f"vis{i}" for i in range(1,5)])
plt.yticks(range(4), [f"txt{i}" for i in range(1,5)])
plt.title("Cosine Similarity Heatmap")
plt.show()
