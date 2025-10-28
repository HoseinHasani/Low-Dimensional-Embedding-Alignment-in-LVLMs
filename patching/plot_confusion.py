import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

mode = "foreground"
# mode = "background"
mode = "FGneighbors"
mode = "just_neighbors"

file_map = {
    "background": "activation_patching_pairs_3to6_and_6to3_background.csv",
    "foreground": "activation_patching_pairs_3to6_and_6to3_foreground.csv",
    "FGneighbors": "activation_patching_pairs_pad1_FGneighbors.csv",
    "just_neighbors": "activation_patching_pairs_pad1_just_neighbors.csv"
}
file_path = file_map[mode]

df = pd.read_csv(file_path, sep=",", engine="python")
df["patches_output"] = pd.to_numeric(df["patches_output"], errors="coerce")

def extract_class(path):
    match = re.search(r'img_(\d+)\.png', str(path))
    return int(match.group(1)) if match else np.nan

df["src_class"] = df["src"].apply(extract_class)
df["tgt_class"] = df["tgt"].apply(extract_class)

def bucket_pred(x):
    if x == 3:
        return "3"
    elif x == 6:
        return "6"
    else:
        return "Other"

df["pred_class"] = df["patches_output"].apply(bucket_pred)

# only keep valid pairs (3→6 and 6→3)
valid_pairs = [(3,6), (6,3)]
df = df[df[["src_class", "tgt_class"]].apply(tuple, axis=1).isin(valid_pairs)]

grouped = (
    df.groupby(["src_class", "tgt_class", "pred_class"])
    .size()
    .unstack(fill_value=0)
    .reindex(pd.MultiIndex.from_tuples(valid_pairs, names=["src_class","tgt_class"]), fill_value=0)
    .reindex(columns=["3","6","Other"], fill_value=0)
)

matrix = grouped.values.astype(float)
matrix = matrix / matrix.sum(axis=1, keepdims=True)

yticklabels = [f"{s} → {t}" for s, t in grouped.index]

plt.figure(figsize=(7,4))
sns.heatmap(
    matrix,
    annot=True,
    cmap="Blues",
    fmt=".2f",
    cbar=True,
    xticklabels=["Pred=3","Pred=6","Pred=Other"],
    yticklabels=yticklabels,
    linewidths=0.5,
    linecolor='gray'
)

plt.title(f"Confusion Matrix ({mode.capitalize()})", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Predicted Class", fontsize=14)
plt.ylabel("Source → Target", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12, rotation=0)
plt.tight_layout()

save_name = f"confusion_matrix_{mode}.png"
plt.savefig(save_name, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure as '{save_name}'")
