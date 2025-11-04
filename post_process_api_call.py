import os
import json
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(
    )

input_dir = "postprocess_results"  
output_dir = "postprocess_refined"
os.makedirs(output_dir, exist_ok=True)

files = [f for f in os.listdir(input_dir) if f.endswith("_fp_marked.json")]

SYSTEM_PROMPT = """You are a text-editing assistant that improves image captions by removing hallucinated objects marked with `$` while keeping the caption fluent and faithful."""

EDITING_PROMPT = """
**Problem Description:**

We are working on a system that generates captions for images. Sometimes, the system may hallucinate or include objects that are not actually present in the image. These hallucinated objects are detected and marked as false positives (FP) using a special token `$` before the object in the caption. For example, a hallucinated object like "$refrigerator" would appear as `$refrigerator`.

**Your Task:**

You are given a caption that includes hallucinated objects marked with `$` (e.g., `$refrigerator`). Your task is to remove **only** the hallucinated objects and keep the rest of the caption intact, maintaining fluency, context, and clarity.

**Strict Instructions:**

1. **Remove Only Hallucinated Objects:**
    - The objects marked with `$` are hallucinated, and you need to **remove only those hallucinated objects** from the caption. For example:
      - "The image shows a spacious studio apartment kitchen with wooden cabinets and $refrigerator." → "The image shows a spacious studio apartment kitchen with wooden cabinets."
      - Do **not** remove any objects in the sentence that are not marked with `$`. These should be kept as they are, since they describe actual objects in the image.
    
2. **Minimal Changes:** 
    - If removing a hallucinated object causes awkward phrasing, make minimal edits to improve the fluency of the sentence. For example:
    - **Do not delete** entire sentence structures unless absolutely necessary to maintain clarity.

3. **Faithfulness to the Original Caption:** 
    - Ensure that the edited caption remains **faithful** to the original context. Do not introduce new details, objects, or replace hallucinated objects with new ones (e.g., don't replace `$refrigerator` with another new object `microwave`).
    - The resulting text should **not lose any original meaning** or introduce new aspects of the scene not present in the image.

4. **Clarity and Brevity:**
    - The edited caption should be clear and concise without being overly terse. Do not over-edit the original content. Make sure that the edited text does not contain objects that marked with $ in the input text.

5. **Output Format:**
    - Provide only the final, edited caption inside **double quotes** (`""`), without any additional text or explanations.


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
            model="gpt-5",  
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=250,
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
