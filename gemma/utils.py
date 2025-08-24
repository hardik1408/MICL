import google.generativeai as genai
import os
import json
import random
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def configure_api(API_KEY):
    if not API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=API_KEY)
    print(" Google AI API configured.")

def load_dataset(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        print(f"Dataset loaded successfully with {len(data)} items.")
        return data
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {path}")
        return None