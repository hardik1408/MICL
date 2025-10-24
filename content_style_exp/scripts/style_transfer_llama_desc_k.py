#!/usr/bin/env python3
"""
style_transfer_llama_k_desc.py

Variant of style_transfer_llama_k.py that, for in-context examples,
passes (content image, style image, textual description of the stylized result)
instead of attaching the resultant example image. The generation task is still
to produce a *description* of the imagined stylized result for the new pair.

Usage:
    python style_transfer_llama_k_desc.py --k 1
    python style_transfer_llama_k_desc.py --k 2
    python style_transfer_llama_k_desc.py             # defaults to k=0 (no examples)

Env (same as original):
    export LLAMA_API_BASE="http://localhost:8000/v1"
    export LLAMA_API_KEY="dummy"
    export LLAMA_MODEL_NAME="your-model-name"
"""

import os
import json
import base64
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple

from openai import OpenAI
from tqdm import tqdm

# ------------------------
# Configuration (unchanged unless noted)
# ------------------------
DATASET_PATH = "content_style_descriptions_300.json"
RESULTS_PATH = "style_transfer_llama_k_desc.json"  # different default out name
CONTENT_PATH_TEMPLATE = "style-transfer-dataset/contents/content_{:d}.jpg"
STYLE_PATH_TEMPLATE = "style-transfer-dataset/styles/style_{:d}.jpg"
MAX_WORKERS = 3
# ------------------------

client = OpenAI(
    base_url=os.environ.get("LLAMA_API_BASE", "http://localhost:8000/v1"),
    api_key=os.environ.get("LLAMA_API_KEY", "skip"),
)

MODEL_NAME = os.environ.get("LLAMA_MODEL_NAME", "meta-llama/Llama-3.2-11B-Vision")


def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    return data


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def choose_examples_for_record(records: List[Dict[str, Any]], current_idx: int, k: int) -> List[Dict[str, Any]]:
    """
    Return up to k records (randomly chosen) that share the same style_number as records[current_idx],
    excluding the current record itself.
    """
    if k <= 0:
        return []

    current = records[current_idx]
    style_num = int(current["style_number"])
    candidates = [r for i, r in enumerate(records) if i != current_idx and int(r["style_number"]) == style_num]
    if not candidates:
        return []

    return random.sample(candidates, min(k, len(candidates)))


def build_prompt_with_examples(record: Dict[str, Any],
                               example_records: List[Dict[str, Any]],
                               content_path: str,
                               style_path: str) -> Tuple[str, List[Tuple[str, str, str]]]:
    """
    Build a prompt that teaches via examples. Each example here is:
      (example_content_path, example_style_path, example_result_description_text)

    Returns (prompt_text, example_triplets)
    """
    # Core teaching prompt
    prompt_parts = [
        "Task: Write a concise, visually grounded description of how a CONTENT image would look after it is "
        "stylized using a STYLE image. Output 2–5 sentences focused only on the imagined stylized result.",
        "",
        "You will first see in-context example(s). Each example consists of:",
        "  • the CONTENT image,",
        "  • the STYLE image, and",
        "  • a short natural-language DESCRIPTION of the resulting stylized image.",
        "",
        "Study how the DESCRIPTION explains: (a) which CONTENT elements remain recognizable (subjects, layout), and "
        "(b) how STYLE attributes (palette, brushwork/texture, lighting, contrast, line quality, patterning) are "
        "applied to those CONTENT elements.",
    ]

    if example_records:
        prompt_parts.append(
            f"There are {len(example_records)} example(s). Pay attention to phrasing that links content elements "
            "to specific style traits (e.g., ‘the city skyline rendered with thick impasto strokes in muted blues’)."
        )

    prompt_parts += [
        "",
        "After the example(s), you will receive a NEW pair: a CONTENT image and the SAME STYLE image.",
        "Your job: Describe the imagined stylized result for the NEW pair. Be specific about the subject, palette, "
        "lighting, texture/brushwork introduced by the STYLE. Do not mention filenames, metadata, "
        "pipeline steps, models, or instructions, only the final visual appearance."
    ]

    prompt = "\n".join(prompt_parts)

    # Build example triplets = (content_img_path, style_img_path, description_text)
    example_triplets: List[Tuple[str, str, str]] = []
    for ex in example_records:
        ex_content_num = int(ex["content_number"])
        ex_style_num = int(ex["style_number"])
        ex_content_path = CONTENT_PATH_TEMPLATE.format(ex_content_num)
        ex_style_path = STYLE_PATH_TEMPLATE.format(ex_style_num)
        ex_desc = (ex.get("image_description") or "").strip()
        if ex_desc:
            example_triplets.append((ex_content_path, ex_style_path, ex_desc))

    # print(example_triplets)
    return prompt, example_triplets


def generate_description(prompt: str,
                         example_triplets: List[Tuple[str, str, str]],
                         content_path: str,
                         style_path: str) -> str:
    """
    Send prompt + images to the multimodal model in this order:
      - prompt text
      - For each example: [example_content_image, example_style_image, example_description_text]
      - Target pair: [content_image, style_image]
    Returns the generated text description.
    """
    content_list: List[Dict[str, Any]] = []
    content_list.append({"type": "text", "text": prompt})

    # Attach example pairs and their textual descriptions
    for ex_content_path, ex_style_path, ex_desc_text in example_triplets:
        try:
            ex_content_b64 = image_to_base64(ex_content_path)
            ex_style_b64 = image_to_base64(ex_style_path)
        except FileNotFoundError:
            # Skip examples with missing files
            continue

        content_list.append({"type": "text", "text": "\nEXAMPLE START\nCONTENT IMAGE:"})
        content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ex_content_b64}"}})
        content_list.append({"type": "text", "text": "STYLE IMAGE:"})
        content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ex_style_b64}"}})
        # Provide the known description as text (this replaces the example resultant image)
        content_list.append({"type": "text", "text": "DESCRIPTION OF THE STYLIZED RESULT:\n" + ex_desc_text})
        content_list.append({"type": "text", "text": "EXAMPLE END\n"})

    # Attach the target content + style images for which we want a new description
    try:
        main_content_b64 = image_to_base64(content_path)
        main_style_b64 = image_to_base64(style_path)
    except FileNotFoundError as e:
        return f"ERROR: missing image file: {e}"

    content_list.append({"type": "text", "text": "\nNEW PAIR — CONTENT IMAGE:"})
    content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{main_content_b64}"}})
    content_list.append({"type": "text", "text": "NEW PAIR — STYLE IMAGE:"})
    content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{main_style_b64}"}})
    content_list.append({"type": "text", "text": "Now produce ONLY the description (2–5 sentences)."})
    
    messages = [{
        "role": "user",
        "content": content_list,
    }]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=500,
        temperature=0.7,
    )

    text = response.choices[0].message.content
    if isinstance(text, list):
        for item in text:
            if isinstance(item, dict) and item.get("type") == "text":
                return (item.get("text") or "").strip()
        return "".join([str(t) for t in text]).strip()
    return (text or "").strip()


def process_record_wrapper(records: List[Dict[str, Any]], idx: int, k: int) -> Dict[str, Any]:
    """
    Process a single record by selecting k examples, building prompt, calling model,
    and returning the result dict to be saved.
    """
    record = records[idx]
    content_number = int(record["content_number"])
    style_number = int(record["style_number"])
    orig_desc = (record.get("image_description") or "").strip()

    content_path = CONTENT_PATH_TEMPLATE.format(content_number)
    style_path = STYLE_PATH_TEMPLATE.format(style_number)

    example_records = choose_examples_for_record(records, idx, k)
    prompt, example_triplets = build_prompt_with_examples(record, example_records, content_path, style_path)

    try:
        generated = generate_description(prompt, example_triplets, content_path, style_path)
    except Exception as e:
        generated = f"ERROR: {e}"

    # Keep the same shape; add which examples were used (optional but handy)
    example_refs = []
    for ex in example_records:
        ex_c = int(ex["content_number"])
        ex_s = int(ex["style_number"])
        example_refs.append(f"content_{ex_c}___style_{ex_s}")

    print(f"Processed Content {content_number}, Style {style_number}, examples={len(example_refs)}")
    return {
        "content_number": content_number,
        "style_number": style_number,
        "image_description": orig_desc,            # ground-truth / provided description from dataset
        "example_refs": example_refs,              # differs from original field name to reflect no image suffix
        "generated_description": generated,        # model output
    }


def main():
    parser = argparse.ArgumentParser(description="Style transfer description generation with in-context examples (k) using example descriptions.")
    parser.add_argument("--k", type=int, default=0, help="Number of in-context examples with the same style to include (default 0).")
    parser.add_argument("--dataset", type=str, default=DATASET_PATH, help="Path to dataset JSON.")
    parser.add_argument("--out", type=str, default=RESULTS_PATH, help="Path to output JSON.")
    parser.add_argument("--limit", type=int, default=0, help="Optional: process only the first N records (0 = all).")
    args = parser.parse_args()

    records = load_dataset(args.dataset)
    if args.limit and args.limit > 0:
        records = records[: args.limit]

    results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_record_wrapper, records, i, max(0, args.k)) for i in range(len(records))]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                results.append(fut.result())
            except Exception as e:
                print("Error processing record:", e)

    results.sort(key=lambda r: (r["content_number"], r["style_number"]))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} results to {args.out}")


if __name__ == "__main__":
    main()
