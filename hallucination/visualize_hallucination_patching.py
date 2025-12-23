import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = "offline_patching_results.csv"
df = pd.read_csv(csv_path)

df["delta"] = (
    df["patched_prob_target_token"]
    - df["clean_prob_target_token"]
)

median_df = (
    df
    .groupby(
        ["src_image_id", "tgt_image_id", "hallucinated_token_idx"],
        as_index=False
    )["delta"]
    .median()
    .rename(columns={"delta": "median_delta"})
)

bins = list(range(5, 156, 10))  # 5–15, 15–25, ..., 145–155
labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins) - 1)]

median_df["idx_bin"] = pd.cut(
    median_df["hallucinated_token_idx"],
    bins=bins,
    labels=labels,
    include_lowest=True,
    right=False
)

bin_stats = (
    median_df
    .groupby("idx_bin")["median_delta"]
    .median()
    .reset_index()
)

bin_stats["idx_bin"] = pd.Categorical(
    bin_stats["idx_bin"],
    categories=labels,
    ordered=True
)
bin_stats = bin_stats.sort_values("idx_bin")

plt.figure(figsize=(13, 5))

plt.bar(
    bin_stats["idx_bin"],
    bin_stats["median_delta"]
)

plt.axhline(0, linestyle="--", linewidth=1)

plt.xlabel("Hallucinated Token Index Bin", fontsize=12)
plt.ylabel("Median (Patched − Clean Probability)", fontsize=12)
plt.title(
    "Median Effect of Patching vs Clean by Token Position",
    fontsize=14
)

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("patched_vs_clean_by_token_bin.png", dpi=300)
plt.show()
