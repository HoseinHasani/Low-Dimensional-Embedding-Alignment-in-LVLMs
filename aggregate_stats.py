import os
import numpy as np
from glob import glob

stats_root = "stats_summary"   # folder created previously
metrics = ["attention", "entropy", "gini"]

position_min = 5
position_max = 160

w_pool = 8      # window size
s_pool = 3      # stride

n_layers = 32
n_heads = 32

save_output_dir = "stats_summary"


# ------------------------------
# helper: sliding average pooling
# ------------------------------
def avg_pool_1d(arr, w, s):
    pooled = []
    N = len(arr)
    for start in range(0, N - w + 1, s):
        window = arr[start:start + w]
        pooled.append(np.mean(window))
    return np.array(pooled)


# ------------------------------
# MAIN AGGREGATION
# ------------------------------
for metric in metrics:

    metric_dir = os.path.join(stats_root, metric)
    assert os.path.isdir(metric_dir), f"Missing folder: {metric_dir}"

    aggregated = {}

    print(f"\nProcessing metric: {metric}")

    for L in range(n_layers):
        for H in range(n_heads):

            fname = os.path.join(metric_dir, f"L_{L}_H_{H}.npy")
            if not os.path.isfile(fname):
                print(f"Warning: Missing file {fname}")
                continue

            # Load dictionary: {pos: logpvalue}
            d = np.load(fname, allow_pickle=True).item()

            # Extract logp values from pos_min to pos_max (in order)
            arr = np.array([d[pos] for pos in range(position_min, position_max + 1)], dtype=float)

            # Step 1 — average pooling
            pooled = avg_pool_1d(arr, w_pool, s_pool)

            # Step 2 — take minimum pooled value → scalar
            scalar = float(np.min(pooled))

            # save in dictionary
            key = f"L_{L}_H_{H}"
            aggregated[key] = scalar

    # Save final aggregated file
    save_path = os.path.join(save_output_dir, f"{metric}_aggregated.npy")
    np.save(save_path, aggregated, allow_pickle=True)

    print(f"Saved aggregated results → {save_path}")

print("\nAll metrics aggregated successfully!")
