import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import seaborn as sns
from scipy.stats import sem, ttest_ind
import json

attention_type = "image"  # Choose between "image" or "text"

data_dir = "data/pope/"
attn_dir = os.path.join(data_dir, "pope_llava_attentions_greedy_all_layers_top20_adversarial")
resp_dir = os.path.join(data_dir, "pope_llava_responses_greedy_all_layers_top20_adversarial")
gt_file = os.path.join(data_dir, "coco/coco_pope_adversarial.json")

n_layers, n_heads = 32, 32
n_top_k = 20
n_select = 20
p_combine_mode = "fisher"  # "fisher" | "min" | "max"
save_dir = f"selected_pope_attention_{attention_type.title()}"
os.makedirs(save_dir, exist_ok=True)
sns.set(style="darkgrid")
eps = 1e-140


def load_ground_truths():
    with open(gt_file, 'r') as f:
        gt_data = [json.loads(line) for line in f]
    return {entry['question_id']: entry['label'].lower() for entry in gt_data}

def load_responses():
    responses = {}
    for response_file in glob(os.path.join(resp_dir, "*.jsonl")):
        with open(response_file, 'r') as f:
            data = json.load(f)
            image_id = int(data['image_id'])
            question_id = data['question_id']
            responses[(image_id, question_id)] = data['caption'].lower()
    return responses

def extract_attention(data_dict):
    entries = data_dict.get("attns", {}).get(attention_type, [])
    if not entries:
        return None
    sub = entries[0]["subtoken_results"][0]
    topk_vals = np.array(sub["topk_values"], dtype=float)
    mean_vals = np.mean(topk_vals[..., :n_top_k], axis=-1)
    return mean_vals

def classify(tp, fp, fn, tn, response, ground_truth, att_vals):
    if att_vals is None:
        return
    if response == 'yes' and ground_truth == 'yes':
        tp.append(att_vals)
    elif response == 'yes' and ground_truth == 'no':
        fp.append(att_vals)
    elif response == 'no' and ground_truth == 'yes':
        fn.append(att_vals)
    elif response == 'no' and ground_truth == 'no':
        tn.append(att_vals)

def aggregate_attention(files):
    tp, fp, fn, tn = [], [], [], []
    ground_truths = load_ground_truths()
    responses = load_responses()

    for f in tqdm(files):
        try:
            with open(f, "rb") as handle:
                data_dict = pickle.load(handle)
        except Exception:
            continue

        qid = int(f.split('_')[-1].split('.')[0])
        img_id = int(f.split('_')[-2])

        response = responses.get((img_id, qid), "")
        gt = ground_truths.get(qid, "")

        att_vals = extract_attention(data_dict)
        classify(tp, fp, fn, tn, response, gt, att_vals)

    return tp, fp, fn, tn

def analyze_layers_heads(tp, fp, fn, tn):
    all_log_pvals = []

    tp_stack = np.stack(tp) if tp else np.zeros((1, n_layers, n_heads))
    fp_stack = np.stack(fp) if fp else np.zeros((1, n_layers, n_heads))
    fn_stack = np.stack(fn) if fn else np.zeros((1, n_layers, n_heads))
    tn_stack = np.stack(tn) if tn else np.zeros((1, n_layers, n_heads))

    for l in range(n_layers):
        for h in range(n_heads):
            tp_vals = tp_stack[:, l, h]
            fp_vals = fp_stack[:, l, h]
            fn_vals = fn_stack[:, l, h]
            tn_vals = tn_stack[:, l, h]

            pvals = []
            if len(tp_vals) > 1:
                pvals.append(ttest_ind(fp_vals, tp_vals, equal_var=False)[1])
            if len(tn_vals) > 1:
                pvals.append(ttest_ind(fp_vals, tn_vals, equal_var=False)[1])
            if len(fn_vals) > 1:
                pvals.append(ttest_ind(fn_vals, tn_vals, equal_var=False)[1])
            if len(fn_vals) > 1:
                pvals.append(ttest_ind(fn_vals, tp_vals, equal_var=False)[1])
                
            if not pvals:
                continue
            if p_combine_mode == "fisher":
                combined = np.exp(-0.5 * -2*np.sum(np.log(np.array(pvals) + eps)))
            elif p_combine_mode == "min":
                combined = min(pvals)
            else:
                combined = max(pvals)
            all_log_pvals.append((l, h, np.log10(combined + eps)))

    selected = sorted(all_log_pvals, key=lambda x: x[2])[:n_select]
    return selected, tp_stack, fp_stack, fn_stack, tn_stack

def plot_selected(selected, tp_stack, fp_stack, fn_stack, tn_stack):
    for l, h, logp in selected:
        tp_vals = tp_stack[:, l, h]
        fp_vals = fp_stack[:, l, h]
        fn_vals = fn_stack[:, l, h]
        tn_vals = tn_stack[:, l, h]

        plt.figure(figsize=(5,4))
        plt.bar([0,1,2,3],
                [tp_vals.mean(), fp_vals.mean(), fn_vals.mean(), tn_vals.mean()],
                yerr=[tp_vals.std(), fp_vals.std(), fn_vals.std(), tn_vals.std()],
                color=["green","red","blue","gray"], capsize=3)
        
        min_ylim = min(tp_vals.mean() - tp_vals.std(), fp_vals.mean() - fp_vals.std(),
                       fn_vals.mean() - fn_vals.std(), tn_vals.mean() - tn_vals.std()) - 4e-5
        max_ylim = max(tp_vals.mean() + tp_vals.std(), fp_vals.mean() + fp_vals.std(),
                       fn_vals.mean() + fn_vals.std(), tn_vals.mean() + tn_vals.std()) + 4e-5

        plt.ylim(min_ylim, max_ylim)
            
        plt.xticks([0,1,2,3], ["TP","FP","FN","TN"])
        plt.ylabel("Attention")
        plt.title(f"L{l+1} H{h+1} log10(pval)={logp:.2f} - {attention_type.title()}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"yesno_L{l}_H{h}_logp{logp:.2f}.png"))
        plt.close()

files = glob(os.path.join(attn_dir, "attentions_*.pkl"))
tp, fp, fn, tn = aggregate_attention(files)
selected, tp_stack, fp_stack, fn_stack, tn_stack = analyze_layers_heads(tp, fp, fn, tn)
plot_selected(selected, tp_stack, fp_stack, fn_stack, tn_stack)
