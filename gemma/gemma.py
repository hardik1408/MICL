import os
import google.generativeai as genai
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('models/gemma-3-27b-it')

# The subject of our NEW image
TARGET_SUBJECT = "A man standing in a field of sunflowers, with a bright blue sky and fluffy white clouds in the background."

# The prompt template for the Qwen-VL model
ART_TEMPLATE = f"""
You are an expert art critic. The following images all share a common artistic style.
Analyze these images and then write a single, detailed paragraph that describes this style.
Focus on color, light, texture, and mood. This paragraph will be used as a prompt for an AI image generator.
Do not mention the specific subjects in the images (like stars, sunflowers, or people). ONLY RETURN THE FINAL PROMPT, NO OTHER TEXT.
Finally, combine your style description with the following subject to create an image. The image should resemble this subject completely: '{TARGET_SUBJECT}'.
"""

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
    ART_TEMPLATE,
    *image_parts
]

response = model.generate_content(contents)

print(response.text)