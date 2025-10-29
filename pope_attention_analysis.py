import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
from glob import glob
from tqdm import tqdm
import seaborn as sns
from collections import defaultdict
from scipy.stats import sem

attention_type = "text"  # Choose between "image" or "text"
data_dir = "data/pope/"
attn_dir = os.path.join(data_dir, "pope_llava_attentions_greedy_all_layers_top20_adversarial")
resp_dir = os.path.join(data_dir, "pope_llava_responses_greedy_all_layers_top20_adversarial")
gt_file = f"{data_dir}coco/coco_pope_adversarial.json"
n_layers, n_heads = 32, 32  
n_top_k = 20  
sns.set(style="darkgrid")

def load_ground_truths():
    with open(gt_file, 'r') as f:
        gt_data = [json.loads(line) for line in f]
    return {entry['question_id']: entry['label'] for entry in gt_data}

def load_responses():
    responses = {}
    for response_file in glob(os.path.join(resp_dir, "*.jsonl")):
        with open(response_file, 'r') as f:
            data = json.load(f)
            image_id = int(data['image_id'])
            question_id = data['question_id']
            responses[(image_id, question_id)] = data['caption']
    return responses

def extract_attention_values(data_dict, attention_type="image"):
    results = []
    entries = data_dict.get("attns", {}).get(attention_type, [])
    for e in entries:
        if not e.get("subtoken_results"):
            continue
        sub = e["subtoken_results"][0]
        topk_vals = np.array(sub["topk_values"], dtype=float)
        if topk_vals.ndim != 3:
            continue
        mean_vals = np.mean(topk_vals[..., :n_top_k], axis=-1)  # Averaging over top-k tokens
        idx = int(sub["idx"])
        results.append((idx, mean_vals))
    return results

def classify_samples(response, ground_truth, attention_values):
    tp, fp, fn, tn = [], [], [], []
    for idx, mean_vals in attention_values:
        if response == 'yes':
            if response == ground_truth:
                tp.append(mean_vals)
            else:
                fp.append(mean_vals)
        elif response == 'no':
            if response != ground_truth:
                fn.append(mean_vals)
            else:
                tn.append(mean_vals)
    return tp, fp, fn, tn

def aggregate_attention_data(files, attention_type="image"):
    tp_collect, fp_collect, fn_collect, tn_collect = [], [], [], []
    ground_truths = load_ground_truths()
    responses = load_responses()

    for file in tqdm(files):
        try:
            with open(file, "rb") as handle:
                data_dict = pickle.load(handle)
        except Exception as e:
            continue
        
        question_id = int(file.split('_')[-1].split('.')[0])
        image_id = int(file.split('_')[-2])
        
        response = responses.get((image_id, question_id), "")
        ground_truth = ground_truths.get(question_id, "")
        
        attention_values = extract_attention_values(data_dict, attention_type)
        
        tp, fp, fn, tn = classify_samples(response.lower(), ground_truth.lower(), attention_values)
        tp_collect.extend(tp)
        fp_collect.extend(fp)
        fn_collect.extend(fn)
        tn_collect.extend(tn)

    return tp_collect, fp_collect, fn_collect, tn_collect

def plot_attention_statistics(tp_data, fp_data, fn_data, tn_data, n_layers, n_heads, attention_type, savepath="pope_attention_statistics.png"):
    fig_w = n_heads * 4
    fig_h = n_layers * 3
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(fig_w, fig_h), sharex=False, sharey=False)
    title = f"Attention Statistics ({attention_type.title()} - TP, FP, FN, TN)"
    fig.suptitle(title, fontsize=60)

    for l in tqdm(range(n_layers)):
        for h in range(n_heads):
            ax = axes[l, h]
            
            tp_vals = np.array([tp_data[l][h] for tp_data in tp_data])
            fp_vals = np.array([fp_data[l][h] for fp_data in fp_data])
            fn_vals = np.array([fn_data[l][h] for fn_data in fn_data])
            tn_vals = np.array([tn_data[l][h] for tn_data in tn_data])
    
            tp_mean, fp_mean, fn_mean, tn_mean = map(np.mean, [tp_vals, fp_vals, fn_vals, tn_vals])
            tp_std, fp_std, fn_std, tn_std = map(np.std, [tp_vals, fp_vals, fn_vals, tn_vals])
    
            ax.bar(0, tp_mean, yerr=tp_std, color="tab:green", label="TP", capsize=5)
            ax.bar(1, fp_mean, yerr=fp_std, color="tab:red", label="FP", capsize=5)
            ax.bar(2, fn_mean, yerr=fn_std, color="tab:blue", label="FN", capsize=5)
            ax.bar(3, tn_mean, yerr=tn_std, color="tab:gray", label="TN", capsize=5)
    
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels(["TP", "FP", "FN", "TN"], fontsize=14)
            ax.set_ylabel("Average Attention Value", fontsize=16)
            ax.set_title(f"Layer {l+1}, Head {h+1}", fontsize=16)
    
            # Independent ylim for this subplot
            local_min = min(tp_mean - tp_std, fp_mean - fp_std, fn_mean - fn_std, tn_mean - tn_std) - 5e-5
            local_max = max(tp_mean + tp_std, fp_mean + fp_std, fn_mean + fn_std, tn_mean + tn_std) + 5e-5
            ax.set_ylim(local_min, local_max)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(savepath)
    plt.show()

files = glob(os.path.join(attn_dir, "attentions_*.pkl"))


tp_data, fp_data, fn_data, tn_data = aggregate_attention_data(files, attention_type)

plot_attention_statistics(tp_data, fp_data, fn_data, tn_data, n_layers, n_heads,
                          attention_type, savepath=f"pope_attention_statistics_{attention_type}.pdf")
