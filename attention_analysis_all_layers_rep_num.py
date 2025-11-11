import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import os
from glob import glob
import seaborn as sns

data_dir = "data/all layers all attention tp fp"
files = glob(os.path.join(data_dir, "attentions_*.pkl"))
n_files = 4000
n_layers, n_heads = 32, 32
avg_win_size = 2
stride_size = 1
n_top_k = 20
n_subtokens = 1
offset_layer = 32 - n_layers
eps = 1e-8
sns.set(style="darkgrid")


# Function to extract attention values along with rep_num
def extract_attention_values(data_dict, cls_, source="image"):
    results = []
    entries = data_dict.get(cls_, {}).get(source, [])
    for e in entries:
        if not e.get("subtoken_results"):
            continue
        rep_num = e["rep_num"]
        for sub in e["subtoken_results"][:n_subtokens]:
            topk_vals = np.array(sub["topk_values"], dtype=float)
            if topk_vals.ndim != 3:
                continue
            idx = int(sub["idx"])
            results.append((idx, topk_vals[..., :n_top_k], rep_num))
    return results

# Aggregating attention values for different rep_num
def aggregate_attention_by_repnum(attention_data, start_token=60, end_token=90):
    # Initialize containers for each category
    tp_rep1 = []
    tp_rep_gt1 = []
    fp_rep1 = []
    fp_rep_gt1 = []

    for idx, topk_vals, rep_num in attention_data:
        if not (start_token <= idx < end_token):
            continue
        # Calculate average attention across the 20 selected tokens
        avg_attention = np.mean(topk_vals, axis=-1)
        
        if rep_num == 1:
            if 'tp' in idx:  # Check for TP category (replace with actual condition)
                tp_rep1.append(avg_attention)
            elif 'fp' in idx:  # Check for FP category (replace with actual condition)
                fp_rep1.append(avg_attention)
        elif rep_num is not None and rep_num > 1:
            if 'tp' in idx:  # Check for TP category (replace with actual condition)
                tp_rep_gt1.append(avg_attention)
            elif 'fp' in idx:  # Check for FP category (replace with actual condition)
                fp_rep_gt1.append(avg_attention)

    return tp_rep1, tp_rep_gt1, fp_rep1, fp_rep_gt1

# Function to plot bar chart for each layer-head
def plot_repnum_attention_bars(tp_rep1, tp_rep_gt1, fp_rep1, fp_rep_gt1, n_layers, n_heads):
    fig_w = n_heads * 4
    fig_h = n_layers * 3
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    fig.suptitle("Average Attention Values by rep_num (TP and FP)", fontsize=60)

    for l in range(n_layers):
        for h in range(n_heads):
            ax = axes[l, h]

            # Aggregate attention values for this layer-head
            tp_rep1_avg = np.mean(tp_rep1, axis=0) if tp_rep1 else 0
            tp_rep_gt1_avg = np.mean(tp_rep_gt1, axis=0) if tp_rep_gt1 else 0
            fp_rep1_avg = np.mean(fp_rep1, axis=0) if fp_rep1 else 0
            fp_rep_gt1_avg = np.mean(fp_rep_gt1, axis=0) if fp_rep_gt1 else 0

            tp_rep1_std = np.std(tp_rep1, axis=0) if tp_rep1 else 0
            tp_rep_gt1_std = np.std(tp_rep_gt1, axis=0) if tp_rep_gt1 else 0
            fp_rep1_std = np.std(fp_rep1, axis=0) if fp_rep1 else 0
            fp_rep_gt1_std = np.std(fp_rep_gt1, axis=0) if fp_rep_gt1 else 0

            # Create bars with error bars
            bar_width = 0.2
            x_pos = np.arange(4)  # 4 categories: TP(rep_num=1), TP(rep_num>1), FP(rep_num=1), FP(rep_num>1)
            
            bars = [
                tp_rep1_avg, tp_rep_gt1_avg, fp_rep1_avg, fp_rep_gt1_avg
            ]
            errors = [
                tp_rep1_std, tp_rep_gt1_std, fp_rep1_std, fp_rep_gt1_std
            ]
            
            ax.bar(x_pos, bars, yerr=errors, width=bar_width, color=["tab:green", "tab:lime", "tab:red", "tab:orange"], alpha=0.7)
            
            # Formatting the plot
            ax.set_xticks(x_pos)
            ax.set_xticklabels(['TP (rep=1)', 'TP (rep>1)', 'FP (rep=1)', 'FP (rep>1)'], fontsize=12)
            ax.set_ylim(0, 1)
            ax.set_ylabel(f"L{l} H{h}", fontsize=14)

            # Optionally, you can add values to the bars
            for i, v in enumerate(bars):
                ax.text(x_pos[i], v + 0.02, f'{v:.2f}', ha='center', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("repnum_attention_bars.pdf")
    plt.show()

# Example usage: Extract attention values and aggregate them
tp_data, fp_data, oth_data = aggregate_across_images(files, n_files)
tp_posmap = aggregate_by_position(tp_data, n_layers, n_heads)
fp_posmap = aggregate_by_position(fp_data, n_layers, n_heads)

# Extract attention values along with rep_num
tp_attention_data = extract_attention_values(tp_data, 'tp')
fp_attention_data = extract_attention_values(fp_data, 'fp')

# Combine both TP and FP data
attention_data = tp_attention_data + fp_attention_data

# Aggregate by rep_num
tp_rep1, tp_rep_gt1, fp_rep1, fp_rep_gt1 = aggregate_attention_by_repnum(attention_data)

# Plot the aggregated bar chart for each layer-head
plot_repnum_attention_bars(tp_rep1, tp_rep_gt1, fp_rep1, fp_rep_gt1, n_layers, n_heads)
