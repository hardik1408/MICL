#!/usr/bin/env python3
"""
style_transfer_deepseek_k.py

In-context example version of style transfer description generation using DeepSeek-VL.

Usage:
    python style_transfer_deepseek_k.py --k 1
    python style_transfer_deepseek_k.py --k 2
    python style_transfer_deepseek_k.py             # defaults to k=0 (no examples)
"""

import os
import json
import argparse
import random
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM

# ------------------------
# Configuration
# ------------------------
DATASET_PATH = "content_style_descriptions_300.json"
RESULTS_PATH = "style_transfer_deepseek_k.json"
CONTENT_PATH_TEMPLATE = "style-transfer-dataset/contents/content_{:d}.jpg"
STYLE_PATH_TEMPLATE = "style-transfer-dataset/styles/style_{:d}.jpg"
EXAMPLE_RESULT_TEMPLATE = "dataset/content_{:d}___style_{:d}___300.jpg"
MAX_WORKERS = 4

# ------------------------
# Load Model
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "deepseek-ai/deepseek-vl-7b-chat"

vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(MODEL_ID)
tokenizer = vl_chat_processor.tokenizer
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)
vl_gpt = vl_gpt.to(torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()

# ------------------------
# Utilities
# ------------------------
def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def choose_examples(records, current_idx, k):
    """Return up to k examples sharing the same style number."""
    if k <= 0:
        return []
    current = records[current_idx]
    style_num = int(current["style_number"])
    candidates = [r for i, r in enumerate(records) if i != current_idx and int(r["style_number"]) == style_num]
    if not candidates:
        return []
    return random.sample(candidates, min(k, len(candidates)))


def build_prompt(record, example_records):
    """Build multimodal prompt for DeepSeek."""
    prompt_parts = [
        "You are given examples showing how a CONTENT image is transformed "
        "into a stylized image by applying the STYLE from another image. "
        "Each example contains three images: content, style, and stylized result."
    ]

    if example_records:
        prompt_parts.append(
            f"There are {len(example_records)} example(s). Observe how style characteristics such as color palette, "
            "lighting, brushwork, and texture transfer from style to content."
        )

    prompt_parts.append(
        "Now, for the new pair of CONTENT and STYLE images, imagine the stylized result. "
        "Describe only the imagined final stylized image in one short paragraph (2–5 sentences). "
        "Focus on subject appearance, color, texture, and artistic feel. Do not mention metadata or filenames."
    )

    guidance = "\n\n".join(prompt_parts)

    # Build DeepSeek-style conversation
    conversation = [
        {"role": "User", "content": guidance},
        {"role": "Assistant", "content": "Understood. Ready for images."},
        {"role": "User", "content": "Now generate the description for the new pair."},
        {"role": "Assistant", "content": ""}
    ]

    return conversation


@torch.inference_mode()
def generate_description(conversation, images):
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=images,
        force_batchify=True,
    ).to(vl_gpt.device)

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=320,
        temperature=0.5,
        do_sample=False,
        use_cache=True,
    )

    text = tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)
    if "Assistant:" in text:
        text = text.split("Assistant:")[-1].strip()
    return text


def process_record(records, idx, k):
    record = records[idx]
    content_number = int(record["content_number"])
    style_number = int(record["style_number"])

    example_records = choose_examples(records, idx, k)
    conversation = build_prompt(record, example_records)

    # Collect all example triplets
    example_images = []
    for ex in example_records:
        ex_content_path = CONTENT_PATH_TEMPLATE.format(int(ex["content_number"]))
        ex_style_path = STYLE_PATH_TEMPLATE.format(int(ex["style_number"]))
        ex_result_path = EXAMPLE_RESULT_TEMPLATE.format(int(ex["content_number"]), int(ex["style_number"]))
        for path in [ex_content_path, ex_style_path, ex_result_path]:
            if os.path.exists(path):
                example_images.append(Image.open(path).convert("RGB"))

    # Add target content + style images
    content_img = Image.open(CONTENT_PATH_TEMPLATE.format(content_number)).convert("RGB")
    style_img = Image.open(STYLE_PATH_TEMPLATE.format(style_number)).convert("RGB")
    all_images = example_images + [content_img, style_img]

    try:
        generated = generate_description(conversation, all_images)
    except Exception as e:
        generated = f"ERROR: {e}"

    return {
        "content_number": content_number,
        "style_number": style_number,
        "image_description": record.get("image_description", ""),
        "generated_description": generated,
        "example_image_names": [
            f"content_{ex['content_number']}___style_{ex['style_number']}___300"
            for ex in example_records
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=0, help="Number of in-context examples")
    parser.add_argument("--dataset", type=str, default=DATASET_PATH)
    parser.add_argument("--out", type=str, default=RESULTS_PATH)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    records = load_dataset(args.dataset)
    if args.limit > 0:
        records = records[:args.limit]

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_record, records, i, args.k) for i in range(len(records))]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                results.append(fut.result())
            except Exception as e:
                print("Error:", e)

    results.sort(key=lambda r: (r["content_number"], r["style_number"]))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} results to {args.out}")


if __name__ == "__main__":
    main()
