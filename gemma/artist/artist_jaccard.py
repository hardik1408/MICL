import google.generativeai as genai
import os
import json
import itertools
from PIL import Image
from tqdm import tqdm # For a nice progress bar
from gemma.utils import *
import argparse
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
load_dotenv()
from templates import ART_TEMPLATE
API_KEY = os.getenv("GOOGLE_API_KEY")

MODEL_NAME = "models/gemma-3-27b-it" 

DATASET_PATH = "picasso.json"


# --- Experiment Parameters ---
K_SHOTS = 2              # Number of in-context examples to provide (can be overwritten by CLI)
THREADS = 2              # Number of threads to use for parallel processing


def generate_vlm_description(prompt_parts, model_name=MODEL_NAME):
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        print(f" An error occurred during API call: {e}")
        return "Error: Could not generate description."


def jaccard_similarity(a: str, b: str) -> float:
    """Simple Jaccard similarity over token sets (lowercased, split on whitespace).
    This is a lightweight retriever for caption similarity.
    """
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def retrieve_top_k(dataset, target_caption, k):
    """Retrieve top-k examples from the dataset whose captions are most similar to target_caption.
    Excludes exact match on caption (i.e., the target itself) by checking equality of caption text.
    Returns a list of dataset items (dictionaries), sorted by descending similarity.
    """
    scores = []
    for item in dataset:
        cand_caption = item.get("caption", "")
        if cand_caption.strip() == "":
            continue
        # exclude if same caption text (assume that's the same item)
        if cand_caption == target_caption:
            continue
        score = jaccard_similarity(cand_caption, target_caption)
        scores.append((score, item))

    # sort by score desc and filter out zero-similarity items first
    scores = sorted(scores, key=lambda x: x[0], reverse=True)
    top_items = [item for score, item in scores if score > 0.0][:k]

    # If not enough similar items found, fall back to the highest-scoring (even zero) items
    if len(top_items) < k:
        extra_needed = k - len(top_items)
        zero_and_rest = [item for score, item in scores if item not in top_items]
        top_items.extend(zero_and_rest[:extra_needed])

    return top_items


def process_item(test_item, dataset, k_shots, model_name=MODEL_NAME):
    """Process a single dataset item: retrieve K shots, build permutations and call the model for each ordering.
    Returns a list of result dicts (possibly empty if errors occur).
    """
    results = []
    test_image_path = test_item.get("image_path")
    test_caption = test_item.get("caption", "")

    # Retrieve K similar examples from entire dataset (excluding the test item by caption equality)
    selected_examples = retrieve_top_k(dataset, test_caption, k_shots)

    # Ensure files exist; drop non-existing images
    selected_examples = [e for e in selected_examples if os.path.exists(e.get("image_path", ""))]
    if not selected_examples:
        # Nothing usable
        return results

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
                failed = True
                break

        if failed:
            continue

        # Target block
        icl_prompt.append("--- TARGET ---\n")
        icl_prompt.append("Now, based on the style of the examples above, generate a new detailed description for an image with the following caption:\n")
        icl_prompt.append(f"CAPTION: {test_caption}\n")
        icl_prompt.append("DETAILED DESCRIPTION:")

        generated_description = generate_vlm_description(icl_prompt, model_name=model_name)
        original_description = test_item.get("image_description")

        results.append({
            "image_path": test_image_path,
            "k_shots_used": n,
            "permutation_index": order_idx - 1,
            "example_paths": examples_path,
            "original_description": original_description,
            "generated_description": generated_description
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Run the VLM experiment with ICL using a retriever and multithreading.")
    parser.add_argument("--dataset", type=str, help="Path to the dataset file.")
    parser.add_argument("--k_shots" , type=int, help="Number of in-context examples to provide.")
    parser.add_argument("--output-dir", type=str, default="generated",help="Directory to save output files.")
    parser.add_argument("--threads", type=int, default=2, help="Number of threads to use.")
    args = parser.parse_args()

    DATASET_PATH = args.dataset
    K_SHOTS = args.k_shots
    OUTPUT_DIR = args.output_dir
    THREADS = args.threads

    configure_api(API_KEY)
    dataset = load_dataset(DATASET_PATH)
    if not dataset:
        return
    print(f"{DATASET_PATH} loaded with {len(dataset)} items.")

    results = []

    # We will process the entire dataset (no train/test split). For each item we retrieve top-k examples
    # from the full dataset (excluding the item itself by caption equality).

    # Use ThreadPoolExecutor to parallelize processing of multiple examples.
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        future_to_item = {executor.submit(process_item, item, dataset, K_SHOTS): item for item in dataset}

        for future in tqdm(as_completed(future_to_item), total=len(future_to_item), desc="Processing examples"):
            try:
                item_results = future.result()
                results.extend(item_results)
            except Exception as e:
                item = future_to_item.get(future)
                print(f"Error processing item {item.get('image_path') if item else '?'}: {e}")

    # Save results
    artist_name = os.path.splitext(os.path.basename(DATASET_PATH))[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = f"{OUTPUT_DIR}/top_{K_SHOTS}_{artist_name}_gemma_jaccard.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n Experiment complete! Results saved to {output_path}")


if __name__ == "__main__":
    main()
