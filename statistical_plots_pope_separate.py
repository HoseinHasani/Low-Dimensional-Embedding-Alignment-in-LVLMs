import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
from glob import glob
from tqdm import tqdm
import seaborn as sns
from scipy.stats import sem, ttest_ind

data_dir = "data/pope/"
attn_dir = os.path.join(data_dir, "pope_llava_attentions_greedy_all_layers_top20_adversarial")
resp_dir = os.path.join(data_dir, "pope_llava_responses_greedy_all_layers_top20_adversarial")
gt_file = os.path.join(data_dir, "coco/coco_pope_adversarial.json")

save_dir = "pope_statistical_heads"
os.makedirs(save_dir, exist_ok=True)

n_layers, n_heads = 32, 32
n_top_k = 20
eps = 1e-50
n_select = 20  
sns.set(style="darkgrid")


def compute_entropy(values):
    """Shannon entropy per layer/head."""
    vals = np.array(values, dtype=float)
    probs = vals / (np.sum(vals, axis=-1, keepdims=True) + eps)
    ent = -np.sum(probs * np.log(probs + eps), axis=-1)
    return ent

def compute_gini(values):
    """Gini coefficient per layer/head."""
    vals = np.array(values, dtype=float)
    probs = vals / (np.sum(vals, axis=-1, keepdims=True) + eps)
    sorted_p = np.sort(probs, axis=-1)
    n = sorted_p.shape[-1]
    coef = 2 * np.arange(1, n + 1) - n - 1
    gini = np.sum(coef * sorted_p, axis=-1) / (n * np.sum(sorted_p, axis=-1) + eps)
    return np.abs(gini)

def load_ground_truths():
    gt = {}
    with open(gt_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            gt[entry["question_id"]] = entry["label"].lower()
    return gt

def load_responses():
    responses = {}
    for path in glob(os.path.join(resp_dir, "*.jsonl")):
        with open(path, "r") as f:
            data = json.load(f)
        image_id = int(data["image_id"])
        qid = data["question_id"]
        responses[(image_id, qid)] = data["caption"].lower()
    return responses

def classify_sample(response, gt):
    if gt == "yes" and response == "yes":
        return "TP"
    elif gt == "yes" and response == "no":
        return "FN"
    elif gt == "no" and response == "yes":
        return "FP"
    elif gt == "no" and response == "no":
        return "TN"
    else:
        return None

def extract_metrics(data_dict):
    """Extract mean attention, entropy, and gini per layer/head."""
    entries = data_dict.get("attns", {}).get("image", [])
    if not entries or not entries[0]["subtoken_results"]:
        return None
    sub = entries[0]["subtoken_results"][0]
    vals = np.array(sub["topk_values"], dtype=float)
    if vals.ndim != 3:
        return None
    vals = vals[..., :n_top_k]
    mean_att = np.mean(vals, axis=-1)
    entropy = compute_entropy(vals)
    gini = compute_gini(vals)
    return mean_att, entropy, gini  # each (32,32)


def aggregate_attention_data(files):
    gt = load_ground_truths()
    responses = load_responses()

    metrics = {
        "TP": {"att": [[[] for _ in range(n_heads)] for _ in range(n_layers)],
               "ent": [[[] for _ in range(n_heads)] for _ in range(n_layers)],
               "gin": [[[] for _ in range(n_heads)] for _ in range(n_layers)]},
        "FP": {"att": [[[] for _ in range(n_heads)] for _ in range(n_layers)],
               "ent": [[[] for _ in range(n_heads)] for _ in range(n_layers)],
               "gin": [[[] for _ in range(n_heads)] for _ in range(n_layers)]},
        "FN": {"att": [[[] for _ in range(n_heads)] for _ in range(n_layers)],
               "ent": [[[] for _ in range(n_heads)] for _ in range(n_layers)],
               "gin": [[[] for _ in range(n_heads)] for _ in range(n_layers)]},
        "TN": {"att": [[[] for _ in range(n_heads)] for _ in range(n_layers)],
               "ent": [[[] for _ in range(n_heads)] for _ in range(n_layers)],
               "gin": [[[] for _ in range(n_heads)] for _ in range(n_layers)]},
    }

    for file in tqdm(files, desc="Aggregating"):
        try:
            with open(file, "rb") as handle:
                data_dict = pickle.load(handle)
        except Exception:
            continue

        try:
            image_id = int(file.split("_")[-2])
            qid = int(file.split("_")[-1].split(".")[0])
        except Exception:
            continue

        response = responses.get((image_id, qid), None)
        gt_label = gt.get(qid, None)
        if response is None or gt_label is None:
            continue

        cat = classify_sample(response, gt_label)
        if cat is None:
            continue

        metrics_triplet = extract_metrics(data_dict)
        if metrics_triplet is None:
            continue

        mean_att, entropy, gini = metrics_triplet
        for l in range(n_layers):
            for h in range(n_heads):
                metrics[cat]["att"][l][h].append(mean_att[l, h])
                metrics[cat]["ent"][l][h].append(entropy[l, h])
                metrics[cat]["gin"][l][h].append(gini[l, h])

    return metrics


def run_statistical_tests(metrics, group_type="yes"):
    """
    group_type='yes' -> compare TP vs FP
    group_type='no'  -> compare TN vs FN
    """
    metric_names = ["att", "ent", "gin"]
    results = []
    for mname in metric_names:
        for l in range(n_layers):
            for h in range(n_heads):
                if group_type == "yes":
                    a = metrics["TP"][mname][l][h]
                    b = metrics["FP"][mname][l][h]
                else:
                    a = metrics["TN"][mname][l][h]
                    b = metrics["FN"][mname][l][h]
                if len(a) < 2 or len(b) < 2:
                    continue
                _, p = ttest_ind(a, b, equal_var=False)
                results.append((mname, l, h, np.log10(p + eps)))
    results.sort(key=lambda x: x[3])  # sort by log(p)
    return results


def plot_selected_heads(metrics, selected, group_type, save_dir):
    for (mname, l, h, logp) in selected:
        if group_type == "yes":
            a = metrics["TP"][mname][l][h]
            b = metrics["FP"][mname][l][h]
            labels = ["TP", "FP"]
            colors = ["tab:green", "tab:red"]
        else:
            a = metrics["TN"][mname][l][h]
            b = metrics["FN"][mname][l][h]
            labels = ["TN", "FN"]
            colors = ["tab:gray", "tab:blue"]

        means = [np.mean(a), np.mean(b)]
        errors = [sem(a) if len(a) > 1 else 0, sem(b) if len(b) > 1 else 0]

        plt.figure(figsize=(4, 4))
        plt.bar(labels, means, yerr=errors, color=colors, capsize=5)
        y_max = np.max(means) + np.max(errors) + 1e-5
        y_min = np.min(means) - np.max(errors) - 1e-5
        plt.ylim(y_min, y_max)
        
        plt.title(f"{group_type.upper()} | {mname.upper()} L{l+1}H{h+1}\nlog10(p)={logp:.2f}")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{group_type}_{mname}_L{l}_H{h}_logp{logp:.1f}.png"), dpi=130)
        plt.close()

files = glob(os.path.join(attn_dir, "attentions_*.pkl"))
metrics = aggregate_attention_data(files)

yes_results = run_statistical_tests(metrics, group_type="yes")
no_results = run_statistical_tests(metrics, group_type="no")

selected_yes = yes_results[:n_select]
selected_no = no_results[:n_select]


plot_selected_heads(metrics, selected_yes, "yes", os.path.join(save_dir, "yes_heads"))
plot_selected_heads(metrics, selected_no, "no", os.path.join(save_dir, "no_heads"))
