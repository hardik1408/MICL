#!/usr/bin/env python3
"""
Run style-transfer description generation with 2-shot multimodal in-context examples.

Requirements:
    pip install openai tqdm

Environment variables (set before running):
    export QWEN_API_BASE="http://localhost:8000/v1"
    export QWEN_API_KEY="dummy"   # or your actual API key
    export QWEN_MODEL_NAME="your-model-name"
"""

import os
import json
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

from openai import OpenAI
from tqdm import tqdm

# ------------------------
# Configuration
# ------------------------
DATASET_PATH = "content_style_descriptions_300.json"
RESULTS_PATH = "style_transfer_qwen_2_shot.json"
CONTENT_PATH_TEMPLATE = "style-transfer-dataset/contents/content_{:d}.jpg"
STYLE_PATH_TEMPLATE = "style-transfer-dataset/styles/style_{:d}.jpg"
MAX_WORKERS = 3
# ------------------------

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="skip"
)

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# True 2-shot examples with content + style images and stylized description
ICL_EXAMPLES = [
    {
        "content_number": 10,
        "style_number": 13,
        "generated_description": (
            "The bridge in the forest is rendered with soft, pastel hues, creating an ethereal quality. "
            "The intricate textures of the trees and foliage are depicted with delicate strokes, giving the scene a dreamy, impressionistic feel. "
            "The light appears diffused, casting a gentle glow that enhances the tranquil atmosphere. "
            "The overall effect is one of serene beauty, as if the bridge is part of a carefully crafted painting rather than a photograph."
        )
    },
    {
        "content_number": 28,
        "style_number": 9,
        "generated_description": (
            "A cluster of delicate white flowers with soft, pastel hues, reminiscent of a dreamlike landscape. "
            "The petals are rendered with a textured, almost painterly quality, capturing their subtle variations in light and shadow. "
            "The background transitions from a pale blue to a hazy white, creating a sense of depth and ethereal atmosphere. "
            "The overall impression is one of tranquility and serenity, achieved through a harmonious blend of color and texture that evokes a gentle, almost wistful mood."
        )
    }
]

def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    return data

def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def build_messages(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = []

    for ex in ICL_EXAMPLES:
        content_path = CONTENT_PATH_TEMPLATE.format(ex["content_number"])
        style_path = STYLE_PATH_TEMPLATE.format(ex["style_number"])
        content_b64 = image_to_base64(content_path)
        style_b64 = image_to_base64(style_path)

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Example: describe the stylized result of the following content-style pair."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{content_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{style_b64}"}},
                {"type": "text", "text": f"Stylized Description: {ex['generated_description']}"}
            ]
        })

    content_b64 = image_to_base64(CONTENT_PATH_TEMPLATE.format(record["content_number"]))
    style_b64 = image_to_base64(STYLE_PATH_TEMPLATE.format(record["style_number"]))

    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": "Now describe the stylized result of the following content-style pair. Only give the final description, no other text."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{content_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{style_b64}"}}
        ]
    })

    return messages

def generate_description(messages: List[Dict[str, Any]]) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=500,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def process_record(record: Dict[str, Any]) -> Dict[str, Any]:
    orig_desc = record.get("image_description", "")
    messages = build_messages(record)
    generated = generate_description(messages)
    print(f"Content {record['content_number']}, Style {record['style_number']}")
    print(generated)
    return {
        "content_number": int(record["content_number"]),
        "style_number": int(record["style_number"]),
        "image_description": orig_desc,
        "generated_description": generated,
    }

def main():
    records = load_dataset(DATASET_PATH)
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_record, rec) for rec in records]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                res = fut.result()
                results.append(res)
            except Exception as e:
                print("Error processing record:", e)

    results.sort(key=lambda r: (r["content_number"], r["style_number"]))

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} results to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
