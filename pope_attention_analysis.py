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

data_dir = "data/pope/"
attn_dir = os.path.join(data_dir, "pope_llava_attentions_greedy_all_layers_top20_adversarial")
resp_dir = os.path.join(data_dir, "pope_llava_responses_greedy_all_layers_top20_adversarial")
gt_file = f"{data_dir}coco/coco_pope_adversarial.json"
n_layers, n_heads = 32, 32  # layers and heads per layer
n_top_k = 20  # We will average over the top-5 values, same as before
sns.set(style="darkgrid")

def load_ground_truths():
    """Load groundtruth labels from coco_pope_adversarial.json."""
    with open(gt_file, 'r') as f:
        gt_data = [json.loads(line) for line in f]
    return {entry['question_id']: entry['label'] for entry in gt_data}

def load_responses():
    """Load responses data (yes/no) from the response folder."""
    responses = {}
    for response_file in glob(os.path.join(resp_dir, "*.jsonl")):
        with open(response_file, 'r') as f:
            data = json.load(f)
            image_id = data['image_id']
            question_id = data['question_id']
            responses[(image_id, question_id)] = data['caption']
    return responses

def extract_attention_values(data_dict):
    """Extract attention values from the data_dict for one image."""
    results = []
    entries = data_dict.get("attns", {}).get("image", [])
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
    """Classify the attention values into TP, FP, FN, TN."""
    tp, fp, fn, tn = [], [], [], []
    for idx, mean_vals in attention_values:
        if response == ground_truth:  # Correct answer
            tp.append(mean_vals)
        else:  # Incorrect answer
            fp.append(mean_vals)
        
        if response != ground_truth:  # False Negative (FN) / False Positive (FP)
            fn.append(mean_vals)  # Assuming if response is incorrect, it belongs to FN
        else:
            tn.append(mean_vals)  # True Negative (TN) if no hallucination occurs
    return tp, fp, fn, tn

def aggregate_attention_data(files):
    """Aggregate attention data and responses for each image."""
    tp_collect, fp_collect, fn_collect, tn_collect = [], [], [], []
    ground_truths = load_ground_truths()
    responses = load_responses()

    for file in tqdm(files):
        try:
            with open(file, "rb") as handle:
                data_dict = pickle.load(handle)
        except Exception as e:
            continue
        
        # Extract the image ID from file name and match it with responses
        image_id = file.split('_')[-1].split('.')[0]
        question_id = int(file.split('_')[-2])
        
        # Get the response and ground truth
        response = responses.get((image_id, question_id), "")
        ground_truth = ground_truths.get(question_id, "")
        
        # Extract attention values
        attention_values = extract_attention_values(data_dict)
        
        # Classify samples into TP, FP, FN, TN
        tp, fp, fn, tn = classify_samples(response, ground_truth, attention_values)
        tp_collect.extend(tp)
        fp_collect.extend(fp)
        fn_collect.extend(fn)
        tn_collect.extend(tn)

    return tp_collect, fp_collect, fn_collect, tn_collect

def plot_attention_statistics(tp_data, fp_data, fn_data, tn_data, n_layers, n_heads, savepath="attention_statistics.png"):
    """Plot the attention statistics with error bars for TP, FP, FN, TN samples."""
    fig_w = n_heads * 4
    fig_h = n_layers * 3
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(fig_w, fig_h), sharex=False, sharey=False)
    fig.suptitle("Attention Statistics (TP, FP, FN, TN)", fontsize=60)

    data_dicts = {"tp": tp_data, "fp": fp_data, "fn": fn_data, "tn": tn_data}
    
    for l in tqdm(range(n_layers)):
        for h in range(n_heads):
            ax = axes[l, h]
            
            # Prepare data for plotting
            tp_vals = np.array([tp_data[l][h] for tp_data in tp_data])
            fp_vals = np.array([fp_data[l][h] for fp_data in fp_data])
            fn_vals = np.array([fn_data[l][h] for fn_data in fn_data])
            tn_vals = np.array([tn_data[l][h] for tn_data in tn_data])

            # Calculate means and standard deviations
            tp_mean = np.mean(tp_vals)
            fp_mean = np.mean(fp_vals)
            fn_mean = np.mean(fn_vals)
            tn_mean = np.mean(tn_vals)

            tp_std = np.std(tp_vals)
            fp_std = np.std(fp_vals)
            fn_std = np.std(fn_vals)
            tn_std = np.std(tn_vals)

            # Plot bars with error bars (standard deviation)
            ax.bar(0, tp_mean, yerr=tp_std, color="tab:green", label="TP", capsize=5)
            ax.bar(1, fp_mean, yerr=fp_std, color="tab:red", label="FP", capsize=5)
            ax.bar(2, fn_mean, yerr=fn_std, color="tab:blue", label="FN", capsize=5)
            ax.bar(3, tn_mean, yerr=tn_std, color="tab:gray", label="TN", capsize=5)

            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels(["TP", "FP", "FN", "TN"], fontsize=14)
            ax.set_ylabel("Average Attention Value", fontsize=16)
            ax.set_ylim(0, 1)
            ax.set_title(f"Layer {l+1}, Head {h+1}", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(savepath)
    plt.show()

files = glob(os.path.join(attn_dir, "attentions_*.pkl"))

tp_data, fp_data, fn_data, tn_data = aggregate_attention_data(files)

plot_attention_statistics(tp_data, fp_data, fn_data, tn_data, n_layers, n_heads, savepath="attention_statistics.pdf")
