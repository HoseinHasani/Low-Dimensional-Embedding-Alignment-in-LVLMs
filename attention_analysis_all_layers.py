import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import seaborn as sns
from scipy.stats import sem
from collections import Counter

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

def extract_attention_values(data_dict, cls_):
    results = []
    entries = data_dict.get(cls_, {}).get("image", [])
    for e in entries:
        if not e.get("subtoken_results"):
            continue
        for sub in e["subtoken_results"][:n_subtokens]:
            topk_vals = np.array(sub["topk_values"], dtype=float)
            if topk_vals.ndim != 3:
                continue
            idx = int(sub["idx"])
            mean_vals = np.mean(topk_vals[..., : n_top_k], axis=-1)
            results.append((idx, mean_vals))
    return results

def aggregate_across_images(files, n_files):
    tp_collect = []
    fp_collect = []
    oth_collect = []
    for f in tqdm(files[:n_files]):
        try:
            with open(f, "rb") as handle:
                data_dict = pickle.load(handle)
        except Exception:
            continue
        tp_collect.extend(extract_attention_values(data_dict, "tp"))
        fp_collect.extend(extract_attention_values(data_dict, "fp"))
        oth_collect.extend(extract_attention_values(data_dict, "other"))
        
    return tp_collect, fp_collect, oth_collect

def aggregate_by_position(attention_data, n_layers, n_heads):
    layer_head_data = {l: {h: {} for h in range(n_heads)} for l in range(n_layers)}
    for idx, mean_vals in attention_data:
        for l in range(n_layers):
            for h in range(n_heads):
                layer_head_data[l][h].setdefault(int(idx), []).append(float(mean_vals[l, h]))
    return layer_head_data

def smooth_with_ci_from_posmap(pos_map, win=3, stride=1):
    if not pos_map:
        return np.array([]), np.array([]), np.array([])
    positions = sorted(pos_map.keys())
    x_arr = np.array(positions)
    xs, ys, cis = [], [], []
    for start in range(0, len(x_arr) - win + 1, stride):
        window_x = x_arr[start:start+win]
        min_x, max_x = int(window_x[0]), int(window_x[-1])
        window_vals = []
        for p in range(min_x, max_x + 1):
            if p in pos_map:
                window_vals.extend(pos_map[p])
        if not window_vals:
            continue
        xs.append(np.mean(window_x))
        ys.append(np.mean(window_vals))
        cis.append(1.96 * sem(window_vals) if len(window_vals) > 1 else 0.0)
    return np.array(xs), np.array(ys), np.array(cis)


def plot_attention_grid(tp_posmap, fp_posmap, oth_posmap, n_layers, n_heads,
                        savepath="attention_grid.pdf", x_min=9, x_max=141):
    fig_w = n_heads * 4
    fig_h = n_layers * 3
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(fig_w, fig_h), sharex=False, sharey=False)
    fig.suptitle("Attention to Image Tokens (TP: green, FP: red, Others: gray)", fontsize=60)
    
    for l in tqdm(range(n_layers)):
        for h in range(n_heads):
            ax = axes[l, h]
            tp_map = tp_posmap[l][h]
            fp_map = fp_posmap[l][h]
            oth_map = oth_posmap[l][h]
            tp_x, tp_y, tp_ci = smooth_with_ci_from_posmap(tp_map, win=avg_win_size, stride=stride_size)
            fp_x, fp_y, fp_ci = smooth_with_ci_from_posmap(fp_map, win=avg_win_size, stride=stride_size)
            oth_x, oth_y, oth_ci = smooth_with_ci_from_posmap(oth_map, win=avg_win_size, stride=stride_size)
            
            if tp_x.size > 0:
                ax.plot(tp_x, tp_y, color="tab:green", linewidth=1.4)
                ax.fill_between(tp_x, tp_y - tp_ci, tp_y + tp_ci, color="tab:blue", alpha=0.2)
            if fp_x.size > 0:
                ax.plot(fp_x, fp_y, color="tab:red", linewidth=1.4)
                ax.fill_between(fp_x, fp_y - fp_ci, fp_y + fp_ci, color="tab:red", alpha=0.2)
            if oth_x.size > 0:
                ax.plot(oth_x, oth_y, color="tab:gray", linewidth=1.4)
                ax.fill_between(oth_x, oth_y - oth_ci, oth_y + oth_ci, color="tab:gray", alpha=0.2)
                
            ax.set_xlim(x_min, x_max)
            
            y_min = min(fp_y[10:-30].min(), tp_y[10:-30].min(), oth_y[10:-30].min())
            y_max = max(fp_y[10:-30].max(), tp_y[10:-30].max(), oth_y[10:-30].max())
            ax.set_ylim(0.97*y_min, 1.03*y_max)
            
            # ys_for_limits = []
            # if tp_y.size > 0:
            #     ys_for_limits.extend((tp_y - 0.1*tp_ci)[6: -30].tolist())
            #     ys_for_limits.extend((tp_y + 0.1*tp_ci)[6: -30].tolist())
            # if fp_y.size > 0:
            #     ys_for_limits.extend((fp_y - 0.1*fp_ci)[4: -30].tolist())
            #     ys_for_limits.extend((fp_y + 0.1*fp_ci)[4: -30].tolist())
            # if ys_for_limits:
            #     y_min = min(ys_for_limits)
            #     y_max = max(ys_for_limits)
            #     if y_max == y_min:
            #         y_min -= 1e-6
            #         y_max += 1e-6
            #     y_pad = 0.05 * (y_max - y_min)
            #     ax.set_ylim(y_min - y_pad, y_max + y_pad)
            # else:
            #     ax.set_ylim(0.0, 1.0)
            
            ax.set_xticks([])
            ax.set_yticks([])
            if l == n_layers - 1:
                ax.set_xlabel(f"H{h}", fontsize=30)
            if h == 0:
                ax.set_ylabel(f"L{l+offset_layer}", fontsize=30)
                
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(savepath)
    plt.show()
    
def get_positions(data_dict):
    positions = []
    for l in range(n_layers):
        for h in range(n_heads):
            positions_ = []
            for key in data_dict[l][h]:
                for j in range(len(data_dict[l][h][key])):
                    positions_.append(key)
            positions.extend(positions_)
    return positions        
    
    
def plot_position_histograms(tp_data, fp_data, oth_data, bins=32):
    tp_positions = get_positions(tp_data)
    fp_positions = get_positions(fp_data)
    oth_positions = get_positions(oth_data)
    
    plt.figure(figsize=(8, 5))
    plt.hist(oth_positions, bins=bins, density=True, alpha=0.6, color="tab:gray", label="Others")
    plt.hist(tp_positions, bins=bins, density=True, alpha=0.6, color="tab:green", label="TP")
    plt.hist(fp_positions, bins=bins, density=True, alpha=0.5, color="tab:red", label="FP")
    
    plt.xlabel("Token Position in Generated Text", fontsize=14)
    plt.xlim(-1, 171)
    plt.ylabel("Normalized Frequency", fontsize=14)
    plt.title("Distribution of Object Token Positions (FP vs TP and Others)", fontsize=15)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("hist_tp_fp_other.png", dpi=140)
    plt.show()
    
    
def plot_token_class_composition(tp_data, fp_data, oth_data, n_layers,
                                 n_heads, bin_size=2,
                                 savepath="token_class_composition.png"):
    

    def count_positions(data_dict):
        pos_counter = Counter()
        for l in range(n_layers):
            for h in range(n_heads):
                for pos in data_dict[l][h].keys():
                    pos_counter[pos] += len(data_dict[l][h][pos])
        return pos_counter

    tp_counts = count_positions(tp_data)
    fp_counts = count_positions(fp_data)
    oth_counts = count_positions(oth_data)

    all_positions = sorted(set(tp_counts.keys()) | set(fp_counts.keys()) | set(oth_counts.keys()))
    if not all_positions:
        print("No position data available to plot.")
        return

    max_pos = max(all_positions)
    binned_positions = np.arange(0, max_pos + bin_size, bin_size)

    tp_perc, fp_perc, oth_perc = [], [], []
    for start in binned_positions:
        end = start + bin_size
        tp_bin = sum(v for k, v in tp_counts.items() if start <= k < end)
        fp_bin = sum(v for k, v in fp_counts.items() if start <= k < end)
        oth_bin = 2*sum(v for k, v in oth_counts.items() if start <= k < end)
        total = tp_bin + fp_bin + oth_bin
        if total == 0:
            tp_perc.append(0)
            fp_perc.append(0)
            oth_perc.append(0)
        else:
            tp_perc.append(tp_bin / total)
            fp_perc.append(fp_bin / total)
            oth_perc.append(oth_bin / total)

    x = np.arange(len(binned_positions))
    bar_width = 1.0  

    plt.figure(figsize=(18, 6))
    plt.bar(x, fp_perc, color="tab:red", width=bar_width, label="FP")
    plt.bar(x, oth_perc, bottom=fp_perc, color="tab:gray", width=bar_width, label="Others", alpha=0.6)
    plt.bar(x, tp_perc, bottom=np.array(fp_perc) + np.array(oth_perc),
            color="tab:green", width=bar_width, label="TP")

    plt.xlabel(f"Token Position (binned, bin size = {bin_size})", fontsize=14)
    plt.ylabel("Percentage of Entries", fontsize=14)
    plt.title("Composition of TP, FP, and Other per Token Position", fontsize=16)
    plt.xlim(-1, 0.81*len(binned_positions))
    plt.ylim(0, 1)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(savepath, dpi=140)
    plt.show()
    
    
def plot_avg_attention_over_layers(tp_posmap, fp_posmap, oth_posmap,
                                   st_layer=0, end_layer=32, n_heads=32,
                                   savepath="avg_attention_over_tokens.png",
                                   avg_win_size=2, stride_size=1):
    """
    Plot averaged attention over heads and summed over layers between st_layer and end_layer.
    """
    def aggregate_class(posmap):
        combined = {}
        for l in range(st_layer, end_layer):
            for h in range(n_heads):
                for idx, vals in posmap[l][h].items():
                    combined.setdefault(idx, []).extend(vals)
        return combined

    tp_combined = aggregate_class(tp_posmap)
    fp_combined = aggregate_class(fp_posmap)
    oth_combined = aggregate_class(oth_posmap)

    def smooth_with_ci(pos_map, win=3, stride=1):
        from scipy.stats import sem
        positions = sorted(pos_map.keys())
        if not positions:
            return np.array([]), np.array([]), np.array([])
        x_arr = np.array(positions)
        xs, ys, cis = [], [], []
        for start in range(0, len(x_arr) - win + 1, stride):
            window_x = x_arr[start:start + win]
            vals = []
            for p in window_x:
                vals.extend(pos_map.get(p, []))
            if not vals:
                continue
            xs.append(np.mean(window_x))
            ys.append(np.mean(vals))
            cis.append(1.96 * sem(vals) if len(vals) > 1 else 0.0)
        return np.array(xs), np.array(ys), np.array(cis)

    tp_x, tp_y, tp_ci = smooth_with_ci(tp_combined, win=avg_win_size, stride=stride_size)
    fp_x, fp_y, fp_ci = smooth_with_ci(fp_combined, win=avg_win_size, stride=stride_size)
    oth_x, oth_y, oth_ci = smooth_with_ci(oth_combined, win=avg_win_size, stride=stride_size)

    plt.figure(figsize=(10, 6))
    if tp_x.size > 0:
        plt.plot(tp_x, tp_y, color="tab:green", label="TP", linewidth=2)
        plt.fill_between(tp_x, tp_y - tp_ci, tp_y + tp_ci, color="tab:green", alpha=0.2)
    if fp_x.size > 0:
        plt.plot(fp_x, fp_y, color="tab:red", label="FP", linewidth=2)
        plt.fill_between(fp_x, fp_y - fp_ci, fp_y + fp_ci, color="tab:red", alpha=0.2)
    # if oth_x.size > 0:
    #     plt.plot(oth_x, oth_y, color="tab:gray", label="Others", linewidth=2)
    #     plt.fill_between(oth_x, oth_y - oth_ci, oth_y + oth_ci, color="tab:gray", alpha=0.2)

    plt.xlabel("Token Index (Generated Text)", fontsize=14)
    plt.ylabel("Aggregated Attention (avg over heads, sum over layers)", fontsize=14)
    plt.title(f"Aggregated Attention ({st_layer}–{end_layer})", fontsize=16)
    plt.legend(fontsize=12)
    plt.xlim(4,156)
    plt.tight_layout()
    plt.savefig(savepath, dpi=130)
    plt.show()


def plot_overall_attention_strength(tp_posmap, fp_posmap, oth_posmap,
                                    st_layer=0, end_layer=15, n_heads=32,
                                    savepath="overall_attention_strength.png"):
    """
    Plot a simple bar chart of overall averaged attention
    (averaged over tokens, heads, and layers) for TP, FP, and Others.
    """
    def compute_overall_mean(posmap):
        vals_all = []
        for l in range(st_layer, end_layer):
            for h in range(n_heads):
                for idx, vals in posmap[l][h].items():
                    vals_all.extend(vals)
        if len(vals_all) == 0:
            return 0.0
        return np.mean(vals_all)

    tp_mean = compute_overall_mean(tp_posmap)
    fp_mean = compute_overall_mean(fp_posmap)
    oth_mean = compute_overall_mean(oth_posmap)

    means = [tp_mean, fp_mean, oth_mean]
    labels = ["True Positives", "False Positives", "Others"]
    colors = ["tab:green", "tab:red", "tab:gray"]

    plt.figure(figsize=(6, 6))
    plt.bar(labels, means, color=colors, alpha=0.8)
    plt.ylabel("Average Attention Strength", fontsize=14)
    plt.title(f"Overall Attention (Layers {st_layer}–{end_layer})", fontsize=15)

    # Annotate values
    for i, val in enumerate(means):
        plt.text(i, val + 0.001, f"{val:.4f}", ha="center", va="bottom", fontsize=12)

    plt.tight_layout()
    plt.savefig(savepath, dpi=130)
    plt.show()




tp_data, fp_data, oth_data = aggregate_across_images(files, n_files)
tp_posmap = aggregate_by_position(tp_data, n_layers, n_heads)
fp_posmap = aggregate_by_position(fp_data, n_layers, n_heads)
oth_posmap = aggregate_by_position(oth_data, n_layers, n_heads)


# plot_attention_grid(tp_posmap, fp_posmap, oth_posmap, n_layers, n_heads, savepath="attention_grid.pdf")

# plot_position_histograms(tp_posmap, fp_posmap, oth_posmap)

# plot_token_class_composition(tp_posmap, fp_posmap, oth_posmap, n_layers, n_heads)

plot_avg_attention_over_layers(tp_posmap, fp_posmap, oth_posmap,
                               st_layer=5, end_layer=19,
                               savepath="avg_attention_L5to19.png")

plot_overall_attention_strength(tp_posmap, fp_posmap, oth_posmap,
                                st_layer=5, end_layer=19,
                                n_heads=32,
                                savepath="overall_attention_strength_L5to19.png")