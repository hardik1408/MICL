#!/usr/bin/env python3
"""
Run style-transfer description generation against a locally hosted
OpenAI-compatible multimodal model server (on port 8000).

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

from openai import OpenAI
from tqdm import tqdm

# ------------------------
# Configuration
# ------------------------
DATASET_PATH = "content_style_descriptions_300.json"
RESULTS_PATH = "style_transfer_llama_testing.json"
CONTENT_PATH_TEMPLATE = "style-transfer-dataset/contents/content_{:d}.jpg"
STYLE_PATH_TEMPLATE = "style-transfer-dataset/styles/style_{:d}.jpg"
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


def build_prompt(record: Dict[str, Any], content_path: str, style_path: str) -> str:
    # orig_desc = record.get("image_description", "")

    prompt = (
        f"You are given two images: one CONTENT image ({content_path}) and one STYLE image ({style_path}). "
        "Imagine the CONTENT iimage has been transformed to adopt the artistic style of the STYLE image. The elements of STYLE image are not applied to the content image, only the artistic style. "
        "Describe only the resulting stylized image in one short paragraph"
        "You do not have to describe the content image or the style image, only the imagined result of combining them. So essentially describe the content of the first image as if it were painted or drawn in the style of the second image. "
        "Focus on the main subject or scene of the image, the artistic treatment, including color palette, lighting, texture and atmosphere. Also comment on how the style influences the perception of the content"
        "lighting. Do not include labels, metadata, or explanations — only the description."
    )

    # if orig_desc:
    #     prompt += f" The content image can be summarized as: \"{orig_desc}\". Keep the description grounded in this."

    return prompt


def generate_description(prompt: str, content_path: str, style_path: str) -> str:
    # Convert both images to base64
    content_b64 = image_to_base64(content_path)
    style_b64 = image_to_base64(style_path)

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{content_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{style_b64}"}},
        ],
    }]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=500,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


def process_record(record: Dict[str, Any]) -> Dict[str, Any]:
    content_number = int(record["content_number"])
    style_number = int(record["style_number"])
    orig_desc = record.get("image_description", "")

    content_path = CONTENT_PATH_TEMPLATE.format(content_number)
    style_path = STYLE_PATH_TEMPLATE.format(style_number)

    prompt = build_prompt(record, content_path, style_path)
    generated = generate_description(prompt, content_path, style_path)
    print(f"Content {content_number}, Style {style_number}")
    print(generated)
    return {
        "content_number": content_number,
        "style_number": style_number,
        "image_description": orig_desc,
        "generated_description": generated,
    }


def main():
    records = load_dataset(DATASET_PATH)
    results = []

    records=records[:10]

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
