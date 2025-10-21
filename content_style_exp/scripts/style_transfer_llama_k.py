#!/usr/bin/env python3
"""
style_transfer_llama_k.py

Extended version of style_transfer_llama.py that supports in-context example
triplets (content, style, resultant) for style transfer description generation.

Usage:
    python style_transfer_llama_k.py --k 1
    python style_transfer_llama_k.py --k 2
    python style_transfer_llama_k.py             # defaults to k=0 (original behavior)

Requirements:
    pip install openai tqdm

Environment variables (set before running):
    export LLAMA_API_BASE="http://localhost:8000/v1"
    export LLAMA_API_KEY="dummy"   # or your actual API key
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
# Configuration (tweak if needed)
# ------------------------
DATASET_PATH = "content_style_descriptions_300.json"
RESULTS_PATH = "style_transfer_llama_testing_k.json"
CONTENT_PATH_TEMPLATE = "style-transfer-dataset/contents/content_{:d}.jpg"
STYLE_PATH_TEMPLATE = "style-transfer-dataset/styles/style_{:d}.jpg"
EXAMPLE_RESULT_TEMPLATE = "dataset/content_{:d}___style_{:d}___300.jpg"  # fixed 300 suffix
MAX_WORKERS = 3
# ------------------------

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="skip"
)

MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision"


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
    # Collect candidates with same style but different content_number
    candidates = [r for i, r in enumerate(records) if i != current_idx and int(r["style_number"]) == style_num]
    if not candidates:
        return []

    k_take = min(k, len(candidates))
    return random.sample(candidates, k_take)


def build_prompt_with_examples(record: Dict[str, Any], example_records: List[Dict[str, Any]], content_path: str, style_path: str) -> Tuple[str, List[Tuple[str, str, str]]]:

    # Detailed prompt header
    prompt_parts = [
        "You are given in-context examples showing how a CONTENT image is transformed "
        "by applying the artistic STYLE from a STYLE image. Each example contains three images:"
        " 1) the content image, 2) the style image, and 3) the resulting stylized image."
    ]

    if example_records:
        prompt_parts.append(f"There are {len(example_records)} example(s). Carefully observe how the style's "
                            "characteristics (color palette, brushwork, texture, lighting, contrast) "
                            "are transferred to the content image in the resulting image. Note which aspects of the "
                            "content are preserved and which artistic attributes of the style dominate the result.")
    # else:
    #     prompt_parts.append("No direct example is provided. Rely on your understanding of style transfer behavior.")

    # Instruction for the task
    prompt_parts.append(
        "Now, after observing the example(s), you will be given a NEW content image and the SAME style image. "
        "Imagine the NEW content image has been transformed to adopt the artistic style of the style image. "
        "Describe only the imagined stylized result in one short paragraph (2-5 sentences). "
        "Focus on: the main subject or scene, color palette, lighting, brushwork/texture, "
        "and how the style changes the perception of the content. Do not include labels, filenames, metadata, or "
        "implementation details — provide only a grounded visual description of the final stylized image."
    )

    prompt = "\n\n".join(prompt_parts)

    # Build list of (example_content_path, example_style_path, example_result_path)
    example_triplets = []
    for ex in example_records:
        ex_content_num = int(ex["content_number"])
        ex_style_num = int(ex["style_number"])
        ex_content_path = CONTENT_PATH_TEMPLATE.format(ex_content_num)
        ex_style_path = STYLE_PATH_TEMPLATE.format(ex_style_num)
        ex_result_path = EXAMPLE_RESULT_TEMPLATE.format(ex_content_num, ex_style_num)
        example_triplets.append((ex_content_path, ex_style_path, ex_result_path))

    return prompt, example_triplets


def generate_description(prompt: str, example_triplets: List[Tuple[str, str, str]], content_path: str, style_path: str) -> str:
    """
    Send prompt + images to the multimodal model in this specific order:
      - For each example: [example_content_image, example_style_image, example_result_image]
      - Then: [content_image, style_image]
    Returns the text response content.
    """
    # Convert main images and examples into base64 and create message content
    content_list = []
    # system or user text
    content_list.append({"type": "text", "text": prompt})

    # Attach example images first (if any)
    for ex_content_path, ex_style_path, ex_result_path in example_triplets:
        # If example files are missing, skip silently (but ideally they exist)
        try:
            ex_content_b64 = image_to_base64(ex_content_path)
            ex_style_b64 = image_to_base64(ex_style_path)
            ex_result_b64 = image_to_base64(ex_result_path)
        except FileNotFoundError:
            # Skip this example if any file is missing
            continue

        content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ex_content_b64}"}})
        content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ex_style_b64}"}})
        content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ex_result_b64}"}})

    # Attach the target content + style images
    main_content_b64 = image_to_base64(content_path)
    main_style_b64 = image_to_base64(style_path)
    content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{main_content_b64}"}})
    content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{main_style_b64}"}})

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

    # Some servers return nested structures; handle the common case:
    text = response.choices[0].message.content
    if isinstance(text, list):
        # Some multimodal endpoints return a list; join or pick first text part
        # try to find a text item
        for item in text:
            if isinstance(item, dict) and item.get("type") == "text":
                return item.get("text", "").strip()
        # fallback
        return "".join([str(t) for t in text]).strip()
    return text.strip()


def process_record_wrapper(records: List[Dict[str, Any]], idx: int, k: int) -> Dict[str, Any]:
    """
    Process a single record by selecting k examples, building prompt, calling model,
    and returning the result dict to be saved.
    """
    record = records[idx]
    content_number = int(record["content_number"])
    style_number = int(record["style_number"])
    orig_desc = record.get("image_description", "")

    content_path = CONTENT_PATH_TEMPLATE.format(content_number)
    style_path = STYLE_PATH_TEMPLATE.format(style_number)

    example_records = choose_examples_for_record(records, idx, k)
    # print(example_records)
    prompt, example_triplets = build_prompt_with_examples(record, example_records, content_path, style_path)

    try:
        generated = generate_description(prompt, example_triplets, content_path, style_path)
    except Exception as e:
        # Return error information in generated_description for debugging
        generated = f"ERROR: {e}"

    # Build example_image_name list of strings like "content_3___style_2___300"
    example_image_names = []
    for ex in example_records:
        ex_content_num = int(ex["content_number"])
        ex_style_num = int(ex["style_number"])
        example_image_names.append(f"content_{ex_content_num}___style_{ex_style_num}___300")

    print(f"Processed Content {content_number}, Style {style_number}, examples={len(example_image_names)}")
    return {
        "content_number": content_number,
        "style_number": style_number,
        "image_description": orig_desc,
        "example_image_name": example_image_names,
        "generated_description": generated,
    }


def main():
    parser = argparse.ArgumentParser(description="Style transfer description generation with in-context examples (k).")
    parser.add_argument("--k", type=int, default=0, help="Number of in-context examples with the same style to include (default 0).")
    parser.add_argument("--dataset", type=str, default=DATASET_PATH, help="Path to dataset JSON.")
    parser.add_argument("--out", type=str, default=RESULTS_PATH, help="Path to output JSON.")
    parser.add_argument("--limit", type=int, default=0, help="Optional: process only the first N records (0 = all).")
    args = parser.parse_args()

    dataset_path = args.dataset
    out_path = args.out
    K = max(0, args.k)

    records = load_dataset(dataset_path)
    if args.limit and args.limit > 0:
        records = records[: args.limit]

    results = []

    # Use thread pool similarly to original
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_record_wrapper, records, i, K) for i in range(len(records))]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                res = fut.result()
                results.append(res)
            except Exception as e:
                print("Error processing record:", e)

    results.sort(key=lambda r: (r["content_number"], r["style_number"]))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
