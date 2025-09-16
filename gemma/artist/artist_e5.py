import google.generativeai as genai
import os
import json
import itertools
from PIL import Image
from tqdm import tqdm
from gemma.utils import *
import argparse
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from sentence_transformers import SentenceTransformer, util
import torch

load_dotenv()
from templates import ART_TEMPLATE

API_KEY = os.getenv("GOOGLE_API_KEY")

MODEL_NAME = "models/gemma-3-27b-it"

DATASET_PATH = "picasso.json"

# --- Experiment Parameters ---
K_SHOTS = 2
THREADS = 1

e5_model = SentenceTransformer("intfloat/e5-base-v2")


def generate_vlm_description(prompt_parts, model_name=MODEL_NAME):
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        print(f"An error occurred during API call: {e}")
        return "Error: Could not generate description."


def build_dense_index(dataset):
    """Precompute E5 embeddings for all captions in the dataset."""
    captions = [item.get("caption", "") for item in dataset]
    passages = [f"passage: {c}" for c in captions]
    embeddings = e5_model.encode(passages, convert_to_tensor=True, normalize_embeddings=True)
    return embeddings


def retrieve_top_k_dense(dataset, dataset_embeddings, target_caption, k):
    """Retrieve top-k examples using E5 embeddings and cosine similarity."""
    if not target_caption.strip():
        return []

    target_embedding = e5_model.encode(
        f"query: {target_caption}", convert_to_tensor=True, normalize_embeddings=True
    )
    scores = util.cos_sim(target_embedding, dataset_embeddings)[0]

    ranked_indices = torch.argsort(scores, descending=True).tolist()

    top_items = []
    for idx in ranked_indices:
        cand_item = dataset[idx]
        cand_caption = cand_item.get("caption", "")
        if cand_caption == target_caption:
            continue
        top_items.append(cand_item)
        if len(top_items) == k:
            break

    return top_items


def process_item(test_item, dataset, dataset_embeddings, k_shots, model_name=MODEL_NAME):
    results = []
    test_image_path = test_item.get("image_path")
    test_caption = test_item.get("caption", "")

    selected_examples = retrieve_top_k_dense(dataset, dataset_embeddings, test_caption, k_shots)

    selected_examples = [e for e in selected_examples if os.path.exists(e.get("image_path", ""))]
    if not selected_examples:
        return results

    n = len(selected_examples)
    example_orders = list(itertools.permutations(selected_examples, n))

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

        icl_prompt.append("--- TARGET ---\n")
        icl_prompt.append(
            "Now, based on the style of the examples above, generate a new detailed description for an image with the following caption:\n"
        )
        icl_prompt.append(f"CAPTION: {test_caption}\n")
        icl_prompt.append("DETAILED DESCRIPTION:")

        generated_description = generate_vlm_description(icl_prompt, model_name=model_name)
        original_description = test_item.get("image_description")

        results.append(
            {
                "image_path": test_image_path,
                "k_shots_used": n,
                "permutation_index": order_idx - 1,
                "example_paths": examples_path,
                "original_description": original_description,
                "generated_description": generated_description,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Run the VLM experiment with E5 dense retriever.")
    parser.add_argument("--dataset", type=str, help="Path to the dataset file.")
    parser.add_argument("--k_shots", type=int, help="Number of in-context examples to provide.")
    parser.add_argument("--output-dir", type=str, default="generated", help="Directory to save output files.")
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

    dataset_embeddings = build_dense_index(dataset)

    results = []

    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        future_to_item = {
            executor.submit(process_item, item, dataset, dataset_embeddings, K_SHOTS): item
            for item in dataset
        }

        for future in tqdm(as_completed(future_to_item), total=len(future_to_item), desc="Processing examples"):
            try:
                item_results = future.result()
                results.extend(item_results)
            except Exception as e:
                item = future_to_item.get(future)
                print(f"Error processing item {item.get('image_path') if item else '?'}: {e}")

    artist_name = os.path.splitext(os.path.basename(DATASET_PATH))[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = f"{OUTPUT_DIR}/top_{K_SHOTS}_{artist_name}_gemma_dense_e5.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nExperiment complete! Results saved to {output_path}")


if __name__ == "__main__":
    main()
