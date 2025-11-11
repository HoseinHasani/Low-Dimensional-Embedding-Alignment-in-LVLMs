import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from scipy.stats import sem
from glob import glob

data_dir = "data/all layers all attention tp fp rep"
n_layers, n_heads = 32, 32  
n_top_k = 20  
n_subtokens = 1  
start_token = 60  
end_token = 90  

n_files = 4000  


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
            idx = int(sub["idx"])  # Token index
            results.append((idx, topk_vals[..., :n_top_k], rep_num))
    return results

def aggregate_attention_by_repnum(attention_data, cls_, start_token=60, end_token=90):
    tp_rep1 = []
    tp_rep_gt1 = []
    fp_rep1 = []
    fp_rep_gt1 = []

    for idx, topk_vals, rep_num in attention_data:
        if not (start_token <= idx < end_token):
            continue
        avg_attention = np.mean(topk_vals, axis=-1)
        
        if cls_ == 'tp':  # For TP samples
            if rep_num == 1:
                tp_rep1.append(avg_attention)
            elif rep_num > 1:
                tp_rep_gt1.append(avg_attention)
        elif cls_ == 'fp':  # For FP samples
            if rep_num == 1:
                fp_rep1.append(avg_attention)
            elif rep_num > 1:
                fp_rep_gt1.append(avg_attention)

    return tp_rep1, tp_rep_gt1, fp_rep1, fp_rep_gt1


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

files = glob(os.path.join(data_dir, "attentions_*.pkl"))

tp_data, fp_data, oth_data = aggregate_across_images(files, n_files)

tp_rep1, tp_rep_gt1, _, _ = aggregate_attention_by_repnum(tp_data, "tp")
_, _, fp_rep1, fp_rep_gt1 = aggregate_attention_by_repnum(fp_data, "fp")




tp_rep1_avg = np.mean(tp_rep1, axis=0) if tp_rep1 else 0
tp_rep_gt1_avg = np.mean(tp_rep_gt1, axis=0) if tp_rep_gt1 else 0
fp_rep1_avg = np.mean(fp_rep1, axis=0) if fp_rep1 else 0
fp_rep_gt1_avg = np.mean(fp_rep_gt1, axis=0) if fp_rep_gt1 else 0

tp_rep1_std = np.std(tp_rep1, axis=0) if tp_rep1 else 0
tp_rep_gt1_std = np.std(tp_rep_gt1, axis=0) if tp_rep_gt1 else 0
fp_rep1_std = np.std(fp_rep1, axis=0) if fp_rep1 else 0
fp_rep_gt1_std = np.std(fp_rep_gt1, axis=0) if fp_rep_gt1 else 0
        

fig_w = 14 * 2
fig_h = 16 * 2
fig, axes = plt.subplots(14, 16, figsize=(fig_w, fig_h), sharex=True, sharey=False)
fig.suptitle("Average Attention Values by rep_num (TP and FP)")


for l in tqdm(range(10, 24)):
    for h in range(16):
        ax = axes[l-12, h]

        bar_width = 0.2
        x_pos = np.arange(4)  # 4 categories: TP(rep_num=1), TP(rep_num>1), FP(rep_num=1), FP(rep_num>1)
        
        bars = [
            tp_rep1_avg[l, h], tp_rep_gt1_avg[l, h], fp_rep1_avg[l, h], fp_rep_gt1_avg[l, h]
        ]
        errors = [
            tp_rep1_std[l, h], tp_rep_gt1_std[l, h], fp_rep1_std[l, h], fp_rep_gt1_std[l, h]
        ]
        
        ax.bar(x_pos, bars, yerr=errors, width=bar_width, color=["tab:green", "tab:blue", "tab:red", "tab:orange"], alpha=0.7)
        
        
        y_min = np.min(np.array(bars) - np.array(errors))
        y_max = np.max(np.array(bars) + np.array(errors))
        
        # ax.set_xticks(x_pos)
        # ax.set_xticklabels(['TP (rep=1)', 'TP (rep>1)', 'FP (rep=1)', 'FP (rep>1)'], fontsize=12)
        ax.set_ylim(y_min, y_max)
        # ax.set_ylabel(f"L{l} H{h}", fontsize=14)


plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("repnum_attention_bars.pdf")
plt.show()