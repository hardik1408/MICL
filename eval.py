from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import json
import os

embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def compute_bleu(reference: str, candidate: str) -> float:
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)


def compute_rouge(reference: str, candidate: str) -> dict:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {metric: scores[metric].fmeasure for metric in scores}


def compute_bertscore(reference: str, candidate: str) -> float:
    P, R, F1 = bert_score([candidate], [reference], lang="en", rescale_with_baseline=True)
    return F1.mean().item()


def compute_cosine_similarity(reference: str, candidate: str) -> float:
    embeddings = embedder.encode([reference, candidate])
    cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])
    return cos_sim[0][0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the VLM experiment with ICL.")
    parser.add_argument("--dataset", type=str, help="Path to the dataset file.")
    parser.add_argument("--output-dir" , type=str, default="eval", help="Directory to save the output file.")
    args = parser.parse_args()
    ref = "A dog is playing with a ball in the park."
    cand = "A puppy plays with a toy outside."
    with open(args.dataset, "r") as f:
        data = json.load(f)
    results = []
    for item in data:
        result = {
            "image_path": item["image_path"],
            "example_path": item["example_path"],
            "BLEU": compute_bleu(item["original_description"], item["generated_description"]),
            "ROUGE": compute_rouge(item["original_description"], item["generated_description"]),
            "BERTScore": compute_bertscore(item["original_description"], item["generated_description"]),
            "Cosine Similarity": float(compute_cosine_similarity(item["original_description"], item["generated_description"]))
        }
        results.append(result)
    file_name = os.path.splitext(os.path.basename(args.dataset))[0]
    output_path = f"{args.output_dir}/eval_{file_name}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print("Results saved")
