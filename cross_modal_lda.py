import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score

use_baseline = False
pca_dim = 10
lda_dim = 4
source_modality = "text"
target_modality = "vision"

vis = np.load("data/just_number_L25/visual_row_means_L25.npy")
txt = np.load("data/just_number_L25/text_fullline_rows_L25.npy")

if use_baseline:
    vis_baseline = np.load("data/just_number_L25/visual_row_means_L25_baseline.npy")
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
labels = []

for n in range(txt.shape[0]):
    for cls in range(txt.shape[1]):
        txt_clean.append(clean_array(txt[n][cls].reshape(1, -1)))
        vis_clean.append(clean_array(vis[n][cls].reshape(1, -1)))
        labels.append(cls)

txt_clean = np.vstack(txt_clean)
vis_clean = np.vstack(vis_clean)
labels = np.array(labels)

scaler = StandardScaler()
txt_z = scaler.fit_transform(txt_clean)
vis_z = scaler.fit_transform(vis_clean)

max_dim = min(txt_z.shape[0], txt_z.shape[1])
pca = PCA(n_components=min(pca_dim, max_dim))
pca.fit(np.vstack([txt_z, vis_z]))

txt_pca = pca.transform(txt_z)
vis_pca = pca.transform(vis_z)

if source_modality == "text":
    X_train, y_train = txt_pca, labels
    X_test, y_test = vis_pca, labels
elif source_modality == "vision":
    X_train, y_train = vis_pca, labels
    X_test, y_test = txt_pca, labels
else:
    raise ValueError("Invalid source_modality")

clf = LDA(n_components=min(lda_dim, len(np.unique(labels))-1))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Source: {source_modality} → Target: {target_modality}")
print(f"Cross-modal LDA classification accuracy: {acc:.4f}")

proj = clf.transform(X_test)
plt.figure(figsize=(6,5))
for cls in np.unique(y_test):
    idx = (y_test == cls)
    plt.scatter(proj[idx, 0], proj[idx, 0]*0, label=f"class {cls}", alpha=0.6)
plt.title(f"LDA projection of {target_modality} (tested)")
plt.legend()
plt.show()
