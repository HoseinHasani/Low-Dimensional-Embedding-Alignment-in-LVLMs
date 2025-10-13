import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

att_dir = "data/all tokens/attentions"
resp_dir = "data/all tokens/responses"
out_dir_img = "visualization_cases_image"
out_dir_txt = "visualization_cases_text"

os.makedirs(out_dir_img, exist_ok=True)
os.makedirs(out_dir_txt, exist_ok=True)

files = glob(os.path.join(att_dir, "attentions_*.pkl"))
colors = {"tp": "green", "fp": "red", "other": "black"}

def extract_attention_values(data_dict, modality):
    att_values = {}
    for cls_ in ["tp", "fp", "other"]:
        entries = data_dict.get(cls_, {}).get(modality, [])
        for e in entries:
            if len(e.get("token_indices", [])) == 0 or len(e.get("subtoken_results", [])) == 0:
                continue
            subtoken_means = []
            for sub in e["subtoken_results"]:
                if "topk_values" in sub and len(sub["topk_values"]) > 0:
                    subtoken_means.append(sub["topk_values"])
            if not subtoken_means:
                continue
            mean_topk_values = np.mean(np.array(subtoken_means, dtype=float), axis=0)
            att_val = float(np.mean(mean_topk_values))
            for idx in e["token_indices"]:
                att_values[idx] = (att_val, cls_)
    return att_values

for f in tqdm(files):
    base = os.path.basename(f)
    image_id = base.split("_")[1].split(".")[0]
    resp_file = os.path.join(resp_dir, f"responses_{image_id}.json")
    if not os.path.exists(resp_file):
        continue
    with open(f, "rb") as handle:
        data_dict = pickle.load(handle)
    with open(resp_file, "r") as h:
        response = json.load(h)
    tokens = response["llava_tokens"]

    for modality, out_dir in [("image", out_dir_img), ("text", out_dir_txt)]:
        att_values = extract_attention_values(data_dict, modality)
        att_list = []
        color_list = []
        for i, tok in enumerate(tokens):
            if i in att_values:
                att_list.append(att_values[i][0])
                color_list.append(colors[att_values[i][1]])
            else:
                att_list.append(0.0)
                color_list.append("gray")
        plt.figure(figsize=(max(8, len(tokens) * 0.15), 4))
        plt.bar(range(len(tokens)), att_list, color=color_list)
        plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=8)
        plt.ylabel("Attention Value", fontsize=12)
        plt.title(f"Image ID {image_id} - {modality.capitalize()} Attention", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{image_id}.png"), dpi=140)
        plt.close()
