import os
from openai import OpenAI
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
import json
import base64

load_dotenv()

client = OpenAI(
    base_url=os.getenv("LLAMA_API_BASE"),
    api_key=os.getenv("LLAMA_API_KEY")
)

artist_name = os.getenv("ARTIST_NAME")

STYLE_DESCRIPTION_PROMPT = """
You are an expert art critic. Carefully analyze the given image and describe it in a single, detailed paragraph focusing on its artistic characteristics.
Discuss:
- Color palette (dominant and secondary tones)
- Lighting qualities (soft, harsh, dramatic, natural, etc.)
- Textures (brushstrokes, smoothness, grain, etc.)
- Mood or atmosphere conveyed
- Composition and how these elements interact to shape the visual experience
Do NOT mention any specific objects, people, or places depicted in the image. Avoid the artist's name, painting title, or broader style categories. Capture the artistic style and feeling, not the content or subject matter.
ONLY RETURN THE FINAL DESCRIPTION, NO EXTRA TEXT.
"""

CAPTION_PROMPT = """
Provide a short, single-sentence caption that plainly summarizes what is shown in the given image, as if for an alt text. Do not use artistic or technical terms—just state the visible content in a neutral way.
ONLY RETURN THE ONE-LINE CAPTION, NO EXTRA TEXT.
"""

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_content(prompt, image_path):
    image_base64 = image_to_base64(image_path)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]
    }]
    response = client.chat.completions.create(
        model=os.getenv("LLAMA_MODEL_NAME"),
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

project_root = Path(__file__).resolve().parent.parent
image_dir = project_root/"dataset"/"artist"/artist_name
image_paths = sorted([str(p) for p in image_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
results = []

for image_path in image_paths:
    if not Path(image_path).exists():
        print(f"Warning: Image not found: {image_path}")
        continue
    
    print(f"Processing: {Path(image_path).name}")
    
    style_description = generate_content(STYLE_DESCRIPTION_PROMPT, image_path)
    caption = generate_content(CAPTION_PROMPT, image_path)
    
    output_image_path = f"dataset/artist/{artist_name}/{Path(image_path).name}"
    results.append({
        "image_path": output_image_path,
        "caption": caption,
        "image_description": style_description
    })

output_dir = project_root/"llama"/artist_name
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir/f"{artist_name}_descriptions_llama.json"

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Done! Descriptions and captions saved in {output_path}")
