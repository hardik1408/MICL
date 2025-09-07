import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_results(path):
    with open(path, "r") as f:
        return json.load(f)

def aggregate_metrics(results):
    metrics = {
        "BLEU": [],
        "ROUGE-1": [],
        "ROUGE-L": [],
        "BERTScore": [],
        "Cosine Similarity": []
    }

    # Collect all values
    for r in results:
        metrics["BLEU"].append(r["BLEU"])
        metrics["ROUGE-1"].append(r["ROUGE"]["rouge1"])
        metrics["ROUGE-L"].append(r["ROUGE"]["rougeL"])
        metrics["BERTScore"].append(r["BERTScore"])
        metrics["Cosine Similarity"].append(r["Cosine Similarity"])

    summary = {}
    for name, values in metrics.items():
        arr = np.array(values)
        summary[name] = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "variance": float(np.var(arr))
        }
    return summary, metrics

def permutation_sensitivity(results):
    """
    For each image, compute min/max per metric across permutations.
    Return aggregated stats across the dataset.
    """
    grouped = {}
    for r in results:
        grouped.setdefault(r["image_path"], []).append(r)

    diffs = { "BLEU": [], "ROUGE-1": [], "ROUGE-L": [], "BERTScore": [], "Cosine Similarity": [] }

    for img, rows in grouped.items():
        for metric in diffs.keys():
            if metric.startswith("ROUGE"):
                vals = [row["ROUGE"]["rouge1"] if metric == "ROUGE-1" else row["ROUGE"]["rougeL"] for row in rows]
            else:
                vals = [row[metric] for row in rows]
            diffs[metric].append(max(vals) - min(vals))

    sensitivity_summary = {}
    for name, values in diffs.items():
        arr = np.array(values)
        sensitivity_summary[name] = {
            "mean_increase": float(np.mean(arr)),
            "max_increase": float(np.max(arr)),
            "median_increase": float(np.median(arr))
        }
    return sensitivity_summary


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results JSON.")
    parser.add_argument("--input", type=str, required=True, help="Path to evaluation JSON file")
    parser.add_argument("--llm", type=str, required=True, help="LLM model name")
    parser.add_argument("--output-prefix", type=str, default="analysis", help="Prefix for plots")
    args = parser.parse_args()

    results = load_results(args.input)

    summary, metrics = aggregate_metrics(results)
    sensitivity = permutation_sensitivity(results)
    input_name = args.input.split("/")[-1].split(".")[0]
    # Save summary and sensitivity to a text file
    with open(f"analysis/{args.llm}/{args.output_prefix}_{input_name}.txt", "w") as f:
        f.write("=== Overall Metrics Summary ===\n")
        for metric, stats in summary.items():
            f.write(f"{metric}: {stats}\n")

        f.write("\n=== Permutation Sensitivity (effect of changing order) ===\n")
        for metric, stats in sensitivity.items():
            f.write(f"{metric}: {stats}\n")

if __name__ == "__main__":
    main()
