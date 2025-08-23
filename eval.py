from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
    ref = "A dog is playing with a ball in the park."
    cand = "A puppy plays with a toy outside."

    print("BLEU:", compute_bleu(ref, cand))
    print("ROUGE:", compute_rouge(ref, cand))
    print("BERTScore:", compute_bertscore(ref, cand))
    print("Cosine Similarity:", compute_cosine_similarity(ref, cand))
