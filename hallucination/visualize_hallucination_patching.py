import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("offline_patching_results.csv")

df["delta"] = df["clean_prob_target_token"] - df["patched_prob_target_token"]

median_df = (
    df
    .groupby(
        ["tgt_image_id", "hallucinated_token_idx"],
        as_index=False
    )["delta"]
    .median()
    .rename(columns={"delta": "median_delta"})
)

bins = list(range(5, 156, 20))
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
    .agg(["median", "std"])
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
    bin_stats["median"],
    yerr=bin_stats["std"],
    capsize=4
)

plt.axhline(0, linestyle="--", linewidth=1)

plt.xlabel("Hallucinated Token Index Bin", fontsize=12)
plt.ylabel("Median (Clean − Patched Probability)", fontsize=12)
plt.title(
    "Median Effect of Clean vs Patched by Token Position",
    fontsize=14
)

plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("clean_minus_patched_by_token_bin.png", dpi=300)
plt.show()
