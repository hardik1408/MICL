import os
import json
import random
import itertools
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # For a nice progress bar
import argparse
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
from templates import ART_TEMPLATE
from llama.utils import *
load_dotenv()

# --- API setup ---
client = OpenAI(
    base_url=os.getenv("LLAMA_API_BASE"),
    api_key=os.getenv("LLAMA_API_KEY")
)

MODEL_NAME = os.getenv("LLAMA_MODEL_NAME")

# --- Experiment Parameters ---
TEST_SET_SIZE = 0.2      # Use 20% of the data for testing
RANDOM_STATE = 42        # For reproducible train/test splits

def generate_vlm_description_llama(prompt_parts, model_name=MODEL_NAME):
    """
    prompt_parts is a list containing text and PIL Images,
    similar to the Gemma setup. We need to convert it into
    Llama's messages format.
    """
    content_blocks = []
    for part in prompt_parts:
        if isinstance(part, str):
            content_blocks.append({"type": "text", "text": part})
        elif isinstance(part, Image.Image):
            # For images, convert to base64 inline
            tmp_path = Path("._tmp_image.jpg")
            part.save(tmp_path, format="JPEG")
            img_b64 = image_to_base64(tmp_path)
            content_blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
            tmp_path.unlink(missing_ok=True)
        else:
            # ignore unknown types
            continue

    messages = [{"role": "user", "content": content_blocks}]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"An error occurred during API call: {e}")
        return "Error: Could not generate description."


def main():
    parser = argparse.ArgumentParser(description="Run the VLM experiment with ICL (Llama).")
    parser.add_argument("--dataset", type=str, help="Path to the dataset file.")
    parser.add_argument("--k_shots", type=int, help="Number of in-context examples to provide.")
    parser.add_argument("--output-dir", type=str, default="generated", help="Directory to save output files.")
    args = parser.parse_args()

    DATASET_PATH = args.dataset
    K_SHOTS = args.k_shots
    OUTPUT_DIR = args.output_dir

    dataset = load_dataset(DATASET_PATH)
    if not dataset:
        return
    print(f"{DATASET_PATH} loaded with {len(dataset)} items.")
    train_data, test_data = train_test_split(
        dataset, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE
    )
    print(f"Dataset split: {len(train_data)} training items, {len(test_data)} testing items.")

    results = []

    for test_item in tqdm(test_data, desc="Processing Test Examples"):
        test_image_path = test_item["image_path"]
        test_caption = test_item["caption"]

        # Choose the shots
        if len(train_data) < K_SHOTS:
            print(f"Warning: Not enough training data ({len(train_data)}) to select {K_SHOTS} shots. Using all available.")
            selected_examples = train_data
        else:
            selected_examples = random.sample(train_data, K_SHOTS)

        selected_examples = [e for e in selected_examples if os.path.exists(e["image_path"])]
        if not selected_examples:
            print("No usable examples found; skipping this test item.")
            continue

        n = len(selected_examples)
        example_orders = list(itertools.permutations(selected_examples, n))
        total_orders = len(example_orders)

        for order_idx, example_order in enumerate(example_orders, start=1):
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
                continue

            # Target block
            icl_prompt.append("--- TARGET ---\n")
            icl_prompt.append("Now, based on the style of the examples above, generate a new detailed description for an image with the following caption:\n")
            icl_prompt.append(f"CAPTION: {test_caption}\n")
            icl_prompt.append("DETAILED DESCRIPTION:")

            generated_description = generate_vlm_description_llama(icl_prompt)
            original_description = test_item.get("image_description")

            print(f"[{order_idx}/{total_orders}] test_image={os.path.basename(test_image_path)}")
            print(generated_description)

            results.append({
                "image_path": test_image_path,
                "k_shots_used": n,
                "permutation_index": order_idx - 1,
                "example_paths": examples_path,
                "original_description": original_description,
                "generated_description": generated_description
            })

    artist_name = os.path.splitext(os.path.basename(args.dataset))[0]
    output_path = f"{OUTPUT_DIR}/top_{args.k_shots}_{artist_name}_llama.json"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nExperiment complete! Results saved to {output_path}")


if __name__ == "__main__":
    main()
