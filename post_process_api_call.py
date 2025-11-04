import os
import json
from tqdm import tqdm
from openai import OpenAI

client = OpenAI()

input_dir = "postprocess_results"  
output_dir = "postprocess_refined"
os.makedirs(output_dir, exist_ok=True)

files = [f for f in os.listdir(input_dir) if f.endswith("_fp_marked.json")]

SYSTEM_PROMPT = """You are a text-editing assistant that improves image captions by removing hallucinated objects marked with `$` while keeping the caption fluent and faithful."""

EDITING_PROMPT = """
**Problem Description:**

We are working on a system that generates captions for images. However, sometimes the system may hallucinate or include objects that are not actually present in the image. These hallucinated objects are detected and marked as false positives (FP) using a special token `$` before the object in the caption. For example, a hallucinated object like "refrigerator" would appear as `$refrigerator`.

The task is to read the caption with the marked FP tokens and remove the hallucinated objects while maintaining the fluency of the text. The goal is to preserve the original context of the image and avoid introducing new objects or altering the intended meaning.

---

**Main Instructions:**

1. **FP Marked Tokens**: Any hallucinated object in the caption is marked with `$` before it (e.g., `$refrigerator`).

2. **Removing Hallucinated Objects**:
   * If the object can be removed without damaging the text or altering its meaning, just remove it.

3. **Minimize Changes**:
   * Make minimal edits and preserve other content.

4. **No New Objects**:
   * Do not introduce new details.

5. **Faithfulness to the Original Caption**:
   * Keep edits faithful to the original meaning.

6. **Clarity and Brevity**:
   * Ensure the caption reads naturally after editing.

7. **No Over-Editing**:
   * Only edit what’s necessary.

8. **Output Format**:
   * Provide only the edited caption inside double quotes (""). Do not add explanations or extra text.

---

The input caption is:
"""

for file in tqdm(files, desc="Processing captions with GPT"):
    path = os.path.join(input_dir, file)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_id = data["image_id"]
    fp_marked_caption = data["fp_marked_caption"]

    user_prompt = f"{EDITING_PROMPT}\n{fp_marked_caption}\n\nEdited caption:"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=200,
        )

        edited_caption = response.choices[0].message.content.strip().strip('"')

    except Exception as e:
        print(f"Error processing {file}: {e}")
        edited_caption = fp_marked_caption  

    output_data = {
        "image_id": image_id,
        "original_caption": data["caption"],
        "fp_marked_caption": fp_marked_caption,
        "edited_caption": edited_caption,
    }

    output_path = os.path.join(output_dir, f"{image_id}_refined.json")
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(output_data, out_f, ensure_ascii=False, indent=4)

print(f"\nFinished refining captions. Saved to: {output_dir}")
