import os
import google.generativeai as genai
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
import json
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('models/gemma-3-27b-it')
from templates import CULTURAL_TEMPLATE
TARGET_SUBJECT = "A man standing in a field of sunflowers, with a bright blue sky and fluffy white clouds in the background."

# The prompt template for the Qwen-VL model


image_paths = [
    "dataset/artist/4.png",
    "dataset/artist/5.png",
]

image_parts = []
for path in image_paths:
    if Path(path).exists():
        image_parts.append(Image.open(path))
    else:
        print(f"Warning: Image path not found: {path}. Skipping.")

contents = [
    CULTURAL_TEMPLATE,
    *image_parts
]

response = model.generate_content(contents)
output = {
    "image_paths": image_paths,
    "target_subject": TARGET_SUBJECT,
    "response": response.text
}

with open("gemma/results/cultural.json", "w") as f:
    json.dump(output, f, indent=2)
print(response.text)