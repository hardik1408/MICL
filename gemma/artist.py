import google.generativeai as genai
import os
import json
import random
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm # For a nice progress bar
from gemma.utils import *
import argparse
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

MODEL_NAME = "models/gemma-3-27b-it" 

DATASET_PATH = "picasso.json"


# --- Experiment Parameters ---
K_SHOTS = 2              # Number of in-context examples to provide
TEST_SET_SIZE = 0.2      # Use 20% of the data for testing
RANDOM_STATE = 42        # For reproducible train/test splits



def generate_vlm_description(prompt_parts, model_name=MODEL_NAME):
    """
    Interacts with the VLM to generate a description.
    
    Args:
        prompt_parts (list): A list containing text strings and PIL Image objects.
        model_name (str): The name of the model to use.
        
    Returns:
        str: The generated text description or an error message.
    """

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        print(f" An error occurred during API call: {e}")
        return "Error: Could not generate description."

def main():
    parser = argparse.ArgumentParser(description="Run the VLM experiment with ICL.")
    parser.add_argument("--dataset", type=str, help="Path to the dataset file.")
    parser.add_argument("--k_shots" , type=int, help="Number of in-context examples to provide.")
    args = parser.parse_args()

    DATASET_PATH = args.dataset
    K_SHOTS = args.k_shots
    configure_api(API_KEY)
    dataset = load_dataset(DATASET_PATH)
    if not dataset:
        return
    print(f"{DATASET_PATH} loaded with {len(dataset)} items.")
    train_data, test_data = train_test_split(
        dataset, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE
    )
    print(f"Dataset split: {len(train_data)} training items, {len(test_data)} testing items.")

    results = []
    
    for test_item in tqdm(test_data, desc=" Processing Test Examples"):
        test_image_path = test_item["image_path"]
        test_caption = test_item["caption"]
        
        if len(train_data) < K_SHOTS:
            print(f"Warning: Not enough training data ({len(train_data)}) to select {K_SHOTS} shots. Using all available.")
            in_context_examples = train_data
        else:
            in_context_examples = random.sample(train_data, K_SHOTS)

        # Build the complex ICL prompt
        icl_prompt = [
            "You are an expert art critic. The following images all share a common artistic style. Analyze these images and then write a single, detailed paragraph that describes this style. Focus on color, light, texture, and mood.\n\n"
        ]
        examples_path = []
        for i, example in enumerate(in_context_examples):
            try:
                icl_prompt.append(f"--- EXAMPLE {i+1} ---\n")
                icl_prompt.append("IMAGE:")
                icl_prompt.append(Image.open(example["image_path"]))
                icl_prompt.append(f"CAPTION: {example['caption']}\n")
                examples_path.append(example["image_path"])
        
            except FileNotFoundError:
                print(f"Skipping example, image not found: {example['image_path']}")
                continue
        
        icl_prompt.append("--- TARGET ---\n")
        icl_prompt.append(f"Now, based on the style of the examples above, generate a new detailed description for an image with the following caption:\n")
        icl_prompt.append(f"CAPTION: {test_caption}\n")
        icl_prompt.append("DETAILED DESCRIPTION:")


        generated_description = generate_vlm_description(icl_prompt)
        original_description = test_item["image_description"]
        print(generated_description)
        results.append({
            "image_path": test_image_path,
            "example_path" : examples_path,
            "original_description": original_description,
            "generated_description": generated_description
        })

    artist_name = os.path.splitext(os.path.basename(args.dataset))[0]
    output_path = f"top_{args.k_shots}_{artist_name}_gemma.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n Experiment complete! Results saved to {output_path}")


if __name__ == "__main__":
    main()