import os
import google.generativeai as genai
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
import json
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('models/gemma-3-27b-it')
from templates import SPATIAL_TEMPLATE
TARGET_SUBJECT = "The cow is ahead of the man on the road"

# The prompt template for the Qwen-VL model


image_paths = [
    "dataset/spatial/1.jpeg",
    "dataset/spatial/2.jpeg",
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
    SPATIAL_TEMPLATE + caption[0] + " " + caption[1],
    *image_parts
]
print(SPATIAL_TEMPLATE)
response = model.generate_content(contents)
output = {
    "image_paths": image_paths,
    "caption": caption,
    "target_subject": TARGET_SUBJECT,
    "response": response.text
}

with open("gemma/results/spatial.json", "w") as f:
    json.dump(output, f, indent=2)
print(response.text)