import json
import numpy as np
import logging
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.ensemble import RandomForestRegressor
import argparse
import csv

# Configura logging
logging.basicConfig(filename='log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_embedding_model(model_choice):
    model_map = {
        "minilm": "all-MiniLM-L6-v2",
        "mpnet": "paraphrase-mpnet-base-v2",
        "distilroberta": "all-distilroberta-v1"
    }
    if model_choice not in model_map:
        raise ValueError("Modello non valido. Scegli tra 'minilm', 'mpnet', 'distilroberta'.")
    model_name = model_map[model_choice]
    logging.info(f"Using embedding model: {model_name}")
    return SentenceTransformer(model_name)

def load_dataset(json_file):
    logging.info(f"Loading dataset from {json_file}")
    with open(json_file, "r") as file:
        data = json.load(file)
    prompts = []
    cwe_mapping = {}
    for cwe, items in data.items():
        for text, score in items.items():
            prompt_data = {"text": text, "score": score, "cwe": cwe}
            prompts.append(prompt_data)
            cwe_mapping[text] = cwe
    logging.info(f"Loaded {len(prompts)} prompts with CWE mappings from dataset")
    return prompts, cwe_mapping

def preprocess_prompts(prompts, model):
    logging.info("Generating embeddings for stored prompts")
    texts = [p["text"] for p in prompts]
    embeddings = model.encode(texts, convert_to_tensor=True)
    logging.info("Embeddings generated successfully")
    return embeddings

def train_model(embeddings, scores):
    logging.info("Training Random Forest model on embeddings")
    X = embeddings.cpu().numpy()
    y = np.array(scores)
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X, y)
    logging.info("Model trained successfully")
    return regressor

def classify_prompt(new_prompt, model, embeddings, prompts, regressor, cwe_mapping, target_cwe=None, top_k=3):
    logging.info(f"Classifying new prompt: {new_prompt}")
    new_embedding = model.encode(new_prompt, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(new_embedding, embeddings)[0]
    sorted_indices = np.argsort(similarities.cpu().numpy())[::-1]
    predicted_probability = regressor.predict([new_embedding.cpu().numpy()])[0]
    candidates = []
    
    for idx in sorted_indices:
        candidate_text = prompts[idx]["text"]
        #candidate_cwe = cwe_mapping.get(candidate_text, "Unknown CWE")
        candidate_cwe = cwe_mapping.get(candidate_text, "Unknown CWE").split('_')[0]
        candidate_similarity = similarities[idx].cpu().item()
        candidates.append((candidate_cwe, candidate_similarity))
    
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]
    total_similarity = sum(sim for _, sim in candidates)
    normalized_candidates = [(cwe, sim / total_similarity if total_similarity > 0 else 0) for cwe, sim in candidates]
    
    target_cwe_prob = None
    if target_cwe:
        for cwe, prob in normalized_candidates:
            if cwe.split('_')[0] == target_cwe:
                target_cwe_prob = prob
                break
        if target_cwe_prob is None:
            target_cwe_prob = 0.0
    
    logging.info(f"Predicted vulnerability probability: {predicted_probability}")
    logging.info(f"Candidate vulnerabilities (CWE, normalized probability): {normalized_candidates}")
    logging.info(f"Probability of being vulnerable to {target_cwe}: {target_cwe_prob}")
    
    return predicted_probability, normalized_candidates, target_cwe_prob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify prompts and find CWE vulnerabilities including target CWE probability."
    )
    parser.add_argument("input_file", type=str, help="Path to the file containing prompts (one per line)")
    parser.add_argument("--model", type=str, default="minilm", help="Embedding model")
    parser.add_argument("--dataset", type=str, default="dataset_secure.json", help="Path to dataset JSON file")
    parser.add_argument("--output", type=str, default="input_prompt.csv", help="Output CSV file")
    parser.add_argument("--topk", type=int, default=3, help="Number of candidate vulnerabilities to return")
    parser.add_argument("--cwe", type=str, default=None, help="Specific CWE to check (e.g., CWE-79)")
    args = parser.parse_args()

    embedding_model = get_embedding_model(args.model)
    prompts, cwe_mapping = load_dataset(args.dataset)
    embeddings = preprocess_prompts(prompts, embedding_model)
    scores = [p["score"] for p in prompts]
    regressor = train_model(embeddings, scores)

    with open(args.input_file, "r") as f:
        new_prompts = [line.strip() for line in f if line.strip()]

    with open(args.output, mode="w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        header = ["prompt", "predicted_probability", "target_cwe_prob",
                  "candidate_cwe_1", "probcwe_1",
                  "candidate_cwe_2", "probcwe_2",
                  "candidate_cwe_3", "probcwe_3"]
        writer.writerow(header)
        
        for prompt in new_prompts:
            predicted_probability, candidates, target_cwe_prob = classify_prompt(
                prompt, embedding_model, embeddings, prompts, regressor, cwe_mapping, target_cwe=args.cwe, top_k=args.topk
            )
            row = [prompt, round(predicted_probability, 2), round(target_cwe_prob, 2) if target_cwe_prob is not None else "None"]
            for candidate in candidates:
                row.extend([candidate[0] if candidate[0] else "None", round(candidate[1], 2)])
            while len(row) < 8:
                row.extend(["None", "None"])
            writer.writerow(row)
            
            print(f"\nPrompt: {prompt}")
            print(f"Predicted Vulnerability Probability: {round(predicted_probability, 2)}")
            print(f"Probability of being vulnerable to {args.cwe}: {round(target_cwe_prob, 2)}")
            print("Candidate Vulnerabilities:")
            for cwe, prob in candidates:
                print(f"  - {cwe}: {round(prob, 2)}")
