#!/usr/bin/env python3
"""
style_transfer_deepseek_desc_k.py

DeepSeek-VL version of style_transfer_llama_desc_k.py:
Uses DeepSeek multimodal inference with in-context examples, where
each example includes (content image, style image, textual description
of stylized result). The model then describes the imagined stylized
result for a new pair.

Usage:
    python style_transfer_deepseek_desc_k.py --k 1
    python style_transfer_deepseek_desc_k.py --k 2
"""

import os
import json
import random
import argparse
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
RESULTS_PATH = "style_transfer_deepseek_desc_k.json"
CONTENT_PATH_TEMPLATE = "style-transfer-dataset/contents/content_{:d}.jpg"
STYLE_PATH_TEMPLATE = "style-transfer-dataset/styles/style_{:d}.jpg"
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
def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def choose_examples(records, current_idx, k):
    """Return up to k examples with the same style_number."""
    if k <= 0:
        return []
    current = records[current_idx]
    style_num = int(current["style_number"])
    candidates = [r for i, r in enumerate(records) if i != current_idx and int(r["style_number"]) == style_num]
    return random.sample(candidates, min(k, len(candidates))) if candidates else []


def build_prompt(record, example_records):
    """
    Build a text prompt describing the task and the in-context structure.
    """
    prompt_parts = [
        "Task: Write a concise, visually grounded description of how a CONTENT image would look after being "
        "stylized using a STYLE image. Output 2–5 sentences focused only on the imagined stylized result.",
        "",
        "You will see example(s). Each example consists of:",
        "  • CONTENT image",
        "  • STYLE image",
        "  • DESCRIPTION of the stylized result",
        "",
        "Observe how the description links content elements to style attributes (color palette, lighting, brushwork, texture, etc.).",
    ]

    if example_records:
        prompt_parts.append(f"There are {len(example_records)} example(s). Study their relationships carefully.")

    prompt_parts.append(
        "After the examples, you will be given a NEW CONTENT image and the SAME STYLE image. "
        "Describe the imagined stylized result for this new pair. Focus on visual appearance: subjects, palette, lighting, texture, and artistic mood."
    )

    prompt = "\n".join(prompt_parts)

    conversation = [
        {"role": "User", "content": prompt},
        {"role": "Assistant", "content": "Understood. Ready for examples."},
        {"role": "User", "content": "Now process the examples and describe the new pair."},
        {"role": "Assistant", "content": ""}
    ]
    return conversation


@torch.inference_mode()
def generate_description(conversation, images):
    """Run the DeepSeek multimodal model for one record."""
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
    )

    text = tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)
    if "Assistant:" in text:
        text = text.split("Assistant:")[-1].strip()
    return text


def process_record(records, idx, k):
    record = records[idx]
    content_number = int(record["content_number"])
    style_number = int(record["style_number"])
    orig_desc = record.get("image_description", "").strip()

    example_records = choose_examples(records, idx, k)
    conversation = build_prompt(record, example_records)

    # Collect images in the same order they are referenced in the prompt:
    # For each example → [content, style]
    # Then → target [content, style]
    all_images = []
    for ex in example_records:
        ex_content_path = CONTENT_PATH_TEMPLATE.format(int(ex["content_number"]))
        ex_style_path = STYLE_PATH_TEMPLATE.format(int(ex["style_number"]))
        for p in [ex_content_path, ex_style_path]:
            if os.path.exists(p):
                all_images.append(Image.open(p).convert("RGB"))
        # Textual example description is part of the conversation prompt, not image input.

    content_img = Image.open(CONTENT_PATH_TEMPLATE.format(content_number)).convert("RGB")
    style_img = Image.open(STYLE_PATH_TEMPLATE.format(style_number)).convert("RGB")
    all_images.extend([content_img, style_img])

    try:
        generated = generate_description(conversation, all_images)
    except Exception as e:
        generated = f"ERROR: {e}"

    return {
        "content_number": content_number,
        "style_number": style_number,
        "image_description": orig_desc,
        "example_refs": [
            f"content_{ex['content_number']}___style_{ex['style_number']}"
            for ex in example_records
        ],
        "generated_description": generated,
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
