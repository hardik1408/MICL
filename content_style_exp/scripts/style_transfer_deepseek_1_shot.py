#!/usr/bin/env python3
import os
import json
from PIL import Image
import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM

DATASET_PATH = "content_style_descriptions_300.json"
RESULTS_PATH = "style_transfer_deepseek_1_shot.json" 
CONTENT_PATH_TEMPLATE = "style-transfer-dataset/contents/content_{:d}.jpg"
STYLE_PATH_TEMPLATE = "style-transfer-dataset/styles/style_{:d}.jpg"

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
    }
]

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "deepseek-ai/deepseek-vl-7b-chat"

vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(MODEL_ID)
tokenizer = vl_chat_processor.tokenizer
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,    
)
vl_gpt = vl_gpt.to(torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]

def img_path(num, is_content=True):
    return (CONTENT_PATH_TEMPLATE if is_content else STYLE_PATH_TEMPLATE).format(int(num))

def build_prompt(record):
    demo_gold = ICL_EXAMPLES[0]["generated_description"].strip()

    guidance = (
        "Describe only the imagined result where the CONTENT image is rendered in the artistic treatment of the STYLE image. "
        "Output exactly one concise paragraph..."
    )

    # Conversation with demo text only, no images
    conversation = [
        {"role": "User", "content": f"{guidance}\n\nExample: produce the stylized description for this pair."},
        {"role": "Assistant", "content": demo_gold},
        {"role": "User", "content": "Now produce only the paragraph for this new pair."},
        {"role": "Assistant", "content": ""}
    ]

    # Load only the target images
    tgt_images = [
        Image.open(img_path(record["content_number"], True)).convert("RGB"),
        Image.open(img_path(record["style_number"], False)).convert("RGB")
    ]

    return conversation, tgt_images

@torch.inference_mode()
def generate_description(conversation, target_images, max_new_tokens=320):
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=target_images,  # only target images
        force_batchify=True,
    ).to(vl_gpt.device)

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        temperature=0.5,
        do_sample=False,
        use_cache=True,
    )
    text = tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)
    if "Assistant:" in text:
        text = text.split("Assistant:")[-1].strip()
    return text

def main():
    records = load_dataset(DATASET_PATH)
    results = []

    for record in tqdm(records, desc="Processing records"):
        conversation, tgt_images = build_prompt(record)
        try:
            generated = generate_description(conversation, tgt_images)
        except Exception as e:
            print(f"Error generating for content {record['content_number']}, style {record['style_number']}: {e}")
            generated = ""
        results.append({
            "content_number": int(record["content_number"]),
            "style_number": int(record["style_number"]),
            "image_description": record.get("image_description", ""),
            "generated_description": generated
        })

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} results to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
