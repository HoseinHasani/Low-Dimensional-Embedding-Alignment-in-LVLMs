import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

AGG_OP = "mean"  # choose from: "median", "mean", "min", "max"

df = pd.read_csv("offline_patching_results.csv")

op_map = {
    "median": np.median,
    "mean": np.mean,
    "min": np.min,
    "max": np.max
}

op = op_map[AGG_OP]

reduced_df = (
    df
    .groupby(
        ["tgt_image_id", "hallucinated_token_idx"],
        as_index=False
    )
    .agg(
        clean_val=("clean_prob_target_token", op),
        patched_val=("patched_prob_target_token", op)
    )
)

reduced_df["delta"] = reduced_df["clean_val"] - reduced_df["patched_val"]

bins = list(range(5, 156, 20))
labels = [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins) - 1)]

reduced_df["idx_bin"] = pd.cut(
    reduced_df["hallucinated_token_idx"],
    bins=bins,
    labels=labels,
    include_lowest=True,
    right=False
)

delta_stats = (
    reduced_df
    .groupby("idx_bin")["delta"]
    .agg(["mean", "std"])
    .reset_index()
)

abs_stats = (
    reduced_df
    .groupby("idx_bin")
    .agg(
        clean_abs_mean=("clean_val", lambda x: np.mean(np.abs(x))),
        clean_abs_std=("clean_val", lambda x: np.std(np.abs(x))),
        patched_abs_mean=("patched_val", lambda x: np.mean(np.abs(x))),
        patched_abs_std=("patched_val", lambda x: np.std(np.abs(x)))
    )
    .reset_index()
)

delta_stats["idx_bin"] = pd.Categorical(
    delta_stats["idx_bin"],
    categories=labels,
    ordered=True
)
delta_stats = delta_stats.sort_values("idx_bin")

abs_stats["idx_bin"] = pd.Categorical(
    abs_stats["idx_bin"],
    categories=labels,
    ordered=True
)
abs_stats = abs_stats.sort_values("idx_bin")

plt.figure(figsize=(13, 5))
plt.bar(
    delta_stats["idx_bin"],
    delta_stats["mean"],
    yerr=delta_stats["std"],
    capsize=4
)
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Hallucinated Token Index Bin", fontsize=12)
plt.ylabel(f"{AGG_OP.capitalize()} Clean − Patched Probability", fontsize=12)
plt.title("Effect of Clean vs Patched by Token Position", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"clean_minus_patched_{AGG_OP}_by_token_bin.png", dpi=300)
plt.show()

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(13, 5))
plt.bar(
    x - width / 2,
    abs_stats["clean_abs_mean"],
    width,
    yerr=abs_stats["clean_abs_std"],
    capsize=4,
    label="Clean"
)
plt.bar(
    x + width / 2,
    abs_stats["patched_abs_mean"],
    width,
    yerr=abs_stats["patched_abs_std"],
    capsize=4,
    label="Patched"
)

plt.xlabel("Hallucinated Token Index Bin", fontsize=12)
plt.ylabel(f"Absolute {AGG_OP.capitalize()} Probability", fontsize=12)
plt.title(f"Absolute Clean vs Patched Token Probabilities ({AGG_OP.capitalize()})", fontsize=14)
plt.xticks(x, labels, rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig(f"absolute_clean_vs_patched_{AGG_OP}_by_token_bin.png", dpi=300)
plt.show()
