import os
import json
from glob import glob
from tqdm import tqdm

# Directories
resp_dir = "data/all tokens/responses"
out_dir_txt = "postprocess_results"

os.makedirs(out_dir_txt, exist_ok=True)

# Get the list of response files
files = glob(os.path.join(resp_dir, "image_id_*.jsonl"))

def mark_fp_tokens(tokens, fp_positions):
    """
    Mark false-positive tokens by prefixing them with '$'.
    FP positions are aligned with llava_tokens (subtokens).
    """
    marked_tokens = []
    for i, tok in enumerate(tokens):
        if i in fp_positions:
            marked_tokens.append(f"${tok}")
        else:
            marked_tokens.append(tok)
    return marked_tokens

def merge_subtokens(tokens):
    """
    Merge subtokens into words, handling prefix markers like '▁'.
    Works directly on possibly FP-marked tokens (with '$' prefix).
    """
    merged_tokens = []
    current_word = ""

    for token in tokens:
        # Handle FP marking before subtoken merging
        is_fp = token.startswith("$")
        token_clean = token[1:] if is_fp else token

        if token_clean.startswith("▁"):  # Start of new word
            if current_word:
                merged_tokens.append(current_word)
            current_word = token_clean[1:]  # Drop ▁
            if is_fp:
                current_word = f"${current_word}"
        else:
            current_word += token_clean  # Continue same word

    if current_word:
        merged_tokens.append(current_word)

    return merged_tokens


# Process each file
for f in tqdm(files):
    base = os.path.basename(f)
    image_id = base.split("_")[1].split(".")[0]

    with open(f, "r", encoding="utf-8") as h:
        lines = h.readlines()

    if not lines:
        continue

    response = json.loads(lines[0])

    tokens = response.get("llava_tokens", [])
    fp_positions = response.get("fp_positions", [])

    if not tokens:
        continue

    marked_tokens = mark_fp_tokens(tokens, fp_positions)

    fp_marked_caption = " ".join(merge_subtokens(marked_tokens))

    final_response = {
        "image_id": image_id,
        "caption": response.get("caption", ""),
        "llava_tokens": tokens,
        "fp_positions": fp_positions,
        "fp_marked_caption": fp_marked_caption
    }

    # Save
    # output_file = os.path.join(out_dir_txt, f"{image_id}_fp_marked.json")
    # with open(output_file, "w", encoding="utf-8") as out_file:
    #     json.dump(final_response, out_file, ensure_ascii=False, indent=4)


    output_file = os.path.join(out_dir_txt, f"{image_id}_fp_marked.jsonl")
    with open(output_file, "w", encoding="utf-8") as out_file:
        json.dump(final_response, out_file, ensure_ascii=False)
        out_file.write("\n")