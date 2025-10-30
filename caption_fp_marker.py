import os
import pickle
import json
import numpy as np
from glob import glob
from tqdm import tqdm


att_dir = "data/all tokens/attentions"
resp_dir = "data/all tokens/responses"
out_dir_txt = "postprocess_results"

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
                    subtoken_means = sub["topk_values"]
                if not subtoken_means:
                    continue
                mean_topk_values = np.mean(np.array(subtoken_means, dtype=float), axis=0)
                idx = sub['idx']
                att_values[idx] = (mean_topk_values, cls_)
    return att_values

def merge_subtokens(tokens):
    merged_tokens = []
    current_word = ""
    for token in tokens:
        if token.startswith("▁"):  # Check for subtoken indicator (Hugging Face standard)
            if current_word:
                merged_tokens.append(current_word)  # Append the previous word if it exists
            current_word = token[1:]  # Remove the "▁" and start a new word
        else:
            current_word += token  # Append subtoken to the current word
    if current_word:
        merged_tokens.append(current_word)  # Add the last word
    return merged_tokens

def mark_fp_in_caption(tokens, att_values):
    # Merge subtokens into words first
    merged_tokens = merge_subtokens(tokens)
    fp_marked_caption = []

    # Iterate over the merged tokens and check for FP tokens
    for i, word in enumerate(merged_tokens):
        # Check if the token is an FP token by looking up the attention values
        if i in att_values and att_values[i][1] == "fp":
            # Add a special token before the FP word (we'll use '$' as the special character)
            fp_marked_caption.append(f"${word}")
        else:
            fp_marked_caption.append(word)

    return " ".join(fp_marked_caption)

# Process each file
for f in tqdm(files):
    base = os.path.basename(f)
    image_id = base.split("_")[1].split(".")[0]
    resp_file = os.path.join(resp_dir, f"image_id_{image_id}.jsonl")
    
    if not os.path.exists(resp_file):
        continue
    
    with open(f, "rb") as handle:
        data_dict = pickle.load(handle)
    
    with open(resp_file, "r", encoding="utf-8") as h:
        lines = h.readlines()
    
    if not lines:
        continue
    
    response = json.loads(lines[0])
    tokens = response["llava_tokens"]

    att_values = extract_attention_values(data_dict, "image")
    
    fp_marked_caption = mark_fp_in_caption(tokens, att_values)
    
    final_response = {
        "image_id": image_id,
        "caption": response["caption"],
        "llava_tokens": response["llava_tokens"],
        "fp_marked_caption": fp_marked_caption
    }

    output_file = os.path.join(out_dir_txt, f"{image_id}_fp_marked.json")
    with open(output_file, "w", encoding="utf-8") as out_file:
        json.dump(final_response, out_file, ensure_ascii=False, indent=4)
