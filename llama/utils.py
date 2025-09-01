import json
import base64
def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    
def load_dataset(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        print(f"Dataset loaded successfully with {len(data)} items.")
        return data
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {path}")
        return None