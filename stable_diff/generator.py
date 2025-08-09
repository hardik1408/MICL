import os
from dotenv import load_dotenv
load_dotenv()
import getpass
from stable_diff.utils import *
import json
STABILITY_KEY = os.getenv("STABILITY_API_KEY")

aspect_ratio = "1:1" #@param ["21:9", "16:9", "3:2", "5:4", "1:1", "4:5", "2:3", "9:16", "9:21"]
seed = 0 #@param {type:"integer"}
output_format = "jpeg" #@param ["jpeg", "png"]

host = f"https://api.stability.ai/v2beta/stable-image/generate/sd3"

# Load prompt from JSON file if available
json_path = "gemma/results/artist_frida.json"
if os.path.exists(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

prompt = data["response"]
params = {
    "prompt" : prompt,
    "aspect_ratio" : aspect_ratio,
    "seed" : seed,
    "output_format" : output_format,
    "model" : "sd3.5-flash"
}

response = send_generation_request(
    host,
    params
)

# Decode response
output_image = response.content
finish_reason = response.headers.get("finish-reason")
seed = response.headers.get("seed")

# Check for NSFW classification
if finish_reason == 'CONTENT_FILTERED':
    raise Warning("Generation failed NSFW classifier")

# Save and display result
generated = f"stable_diff/generated_images/gemma/generated_{seed}.{output_format}"
with open(generated, "wb") as f:
    f.write(output_image)
print(f"Saved image {generated}")


