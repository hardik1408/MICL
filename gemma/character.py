import os
import google.generativeai as genai
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
import json
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('models/gemma-3-27b-it')
from templates import CHARACTER_TEMPLATE
TARGET_SUBJECT = "a bustling city street during the evening rush hour"

# The prompt template for the Qwen-VL model


image_paths = [
    "dataset/character/1.jpeg",
    "dataset/character/2.jpeg",
]

caption = [
    "The cat is behind the laptop.",
    "The cake is at the edge of the dining table."
]

image_parts = []
for path in image_paths:
    if Path(path).exists():
        image_parts.append(Image.open(path))
    else:
        print(f"Warning: Image path not found: {path}. Skipping.")

contents = [
    CHARACTER_TEMPLATE,
    *image_parts
]
print(CHARACTER_TEMPLATE)
response = model.generate_content(contents)
output = {
    "image_paths": image_paths,
    "caption": caption,
    "target_subject": TARGET_SUBJECT,
    "response": response.text
}

with open("gemma/results/character.json", "w") as f:
    json.dump(output, f, indent=2)
print(response.text)