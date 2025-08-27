import google.generativeai as genai
import os
import json
import random
import itertools
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm # For a nice progress bar
from gemma.utils import *
import argparse
from dotenv import load_dotenv
load_dotenv()
from templates import ART_TEMPLATE
API_KEY = os.getenv("GOOGLE_API_KEY")

MODEL_NAME = "models/gemma-3-27b-it" 

DATASET_PATH = "picasso.json"


# --- Experiment Parameters ---
K_SHOTS = 2              # Number of in-context examples to provide
TEST_SET_SIZE = 0.2      # Use 20% of the data for testing
RANDOM_STATE = 42        # For reproducible train/test splits



def generate_vlm_description(prompt_parts, model_name=MODEL_NAME):
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
    parser.add_argument("--output-dir", type=str, default="generated",help="Directory to save output files.")
    args = parser.parse_args()

    DATASET_PATH = args.dataset
    K_SHOTS = args.k_shots
    OUTPUT_DIR = args.output_dir

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

        # Choose the shots
        if len(train_data) < K_SHOTS:
            print(f"Warning: Not enough training data ({len(train_data)}) to select {K_SHOTS} shots. Using all available.")
            selected_examples = train_data
        else:
            selected_examples = random.sample(train_data, K_SHOTS)

        # Optionally ensure the selected examples actually exist on disk;
        # if not, drop them (or you could resample to keep K)
        selected_examples = [e for e in selected_examples if os.path.exists(e["image_path"])]
        if not selected_examples:
            print("No usable examples found; skipping this test item.")
            continue

        n = len(selected_examples)

        # Build permutations of the chosen set; if n==1 you'll just get one ordering
        example_orders = list(itertools.permutations(selected_examples, n))
        total_orders = len(example_orders)  # factorial(n)

        for order_idx, example_order in enumerate(example_orders, start=1):
            # Build a fresh prompt for THIS permutation only
            icl_prompt = [ART_TEMPLATE]
            examples_path = []
            failed = False

            for i, example in enumerate(example_order):
                try:
                    icl_prompt.append(f"--- EXAMPLE {i+1} ---\n")
                    icl_prompt.append("IMAGE:")
                    icl_prompt.append(Image.open(example["image_path"]))
                    icl_prompt.append(f"CAPTION: {example['caption']}\n")
                    examples_path.append(example["image_path"])
                except FileNotFoundError:
                    print(f"Skipping permutation {order_idx}/{total_orders}: image not found {example['image_path']}")
                    failed = True
                    break

            if failed:
                continue  # do not query VLM with a partial permutation

            # Target block
            icl_prompt.append("--- TARGET ---\n")
            icl_prompt.append("Now, based on the style of the examples above, generate a new detailed description for an image with the following caption:\n")
            icl_prompt.append(f"CAPTION: {test_caption}\n")
            icl_prompt.append("DETAILED DESCRIPTION:")

            # Query VLM ONCE for this ordering
            generated_description = generate_vlm_description(icl_prompt)
            original_description = test_item.get("image_description")

            print(f"[{order_idx}/{total_orders}] test_image={os.path.basename(test_image_path)}")
            print(generated_description)

            results.append({
                "image_path": test_image_path,
                "k_shots_used": n,
                "permutation_index": order_idx - 1,       # zero-based index if you need it
                "example_paths": examples_path,
                "original_description": original_description,
                "generated_description": generated_description
            })


    artist_name = os.path.splitext(os.path.basename(args.dataset))[0]
    output_path = f"{OUTPUT_DIR}/top_{args.k_shots}_{artist_name}_gemma.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n Experiment complete! Results saved to {output_path}")


if __name__ == "__main__":
    main()