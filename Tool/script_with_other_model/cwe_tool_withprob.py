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

# Funzione per selezionare il modello di embedding in base alla scelta dell'utente
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

# Carica dataset di prompt e associazione CWE
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

# Genera embeddings per i prompt (in batch)
def preprocess_prompts(prompts, model):
    logging.info("Generating embeddings for stored prompts")
    texts = [p["text"] for p in prompts]
    embeddings = model.encode(texts, convert_to_tensor=True)
    logging.info("Embeddings generated successfully")
    return embeddings

# Addestra il modello di regressione
def train_model(embeddings, scores):
    logging.info("Training Random Forest model on embeddings")
    X = embeddings.cpu().numpy()
    y = np.array(scores)
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X, y)
    logging.info("Model trained successfully")
    return regressor

# Deduplica le candidate CWE, mantenendo solo quella con la maggiore similarità per ogni CWE.
def deduplicate_candidates(candidates):
    dedup = {}
    for cwe, similarity in candidates:
        if cwe in dedup:
            if similarity > dedup[cwe]:
                dedup[cwe] = similarity
        else:
            dedup[cwe] = similarity
    # Converti in lista di tuple
    return [(cwe, sim) for cwe, sim in dedup.items()]

# Classifica un nuovo prompt e restituisce la probabilità complessiva e fino a 3 candidate CWE.
def classify_prompt(new_prompt, model, embeddings, prompts, regressor, cwe_mapping, top_k=3):
    logging.info(f"Classifying new prompt: {new_prompt}")
    new_embedding = model.encode(new_prompt, convert_to_tensor=True)
    
    # Calcola la similarità coseno tra il nuovo embedding e quelli del dataset
    similarities = util.pytorch_cos_sim(new_embedding, embeddings)[0]
    sorted_indices = np.argsort(similarities.cpu().numpy())[::-1]
    
    # Probabilità complessiva predetta per il nuovo prompt (score globale)
    predicted_probability = regressor.predict([new_embedding.cpu().numpy()])[0]
    
    # Costruisci la lista dei candidati: (CWE troncata, similarità)
    candidates = []
    for idx in sorted_indices:
        candidate_text = prompts[idx]["text"]
        candidate_cwe = cwe_mapping.get(candidate_text, "Unknown CWE").split('_')[0]
        candidate_similarity = similarities[idx].cpu().item()
        candidates.append((candidate_cwe, candidate_similarity))
    
    # Rimuovi duplicati per CWE, mantenendo solo quello con maggiore similarità
    candidates = deduplicate_candidates(candidates)
    # Ordina per similarità decrescente
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    
    # Seleziona le top_k candidate
    candidates = candidates[:top_k]
    
    # Calcola la somma totale delle similarità delle candidate selezionate
    total_similarity = sum(sim for _, sim in candidates)
    normalized_candidates = []
    for candidate in candidates:
        cwe, sim = candidate
        # Normalizza la similarità per ottenere una "probabilità" per quella candidate
        prob = sim / total_similarity if total_similarity > 0 else 0
        normalized_candidates.append((cwe, prob))
    
    # Se ci sono meno di top_k candidate, aggiungi tuple con None
    while len(normalized_candidates) < top_k:
        normalized_candidates.append((None, None))
    
    logging.info(f"Predicted vulnerability probability: {predicted_probability}")
    logging.info(f"Candidate vulnerabilities (CWE, normalized probability): {normalized_candidates}")
    return predicted_probability, normalized_candidates

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify prompts and find CWE vulnerabilities using different Sentence-Transformers models."
    )
    parser.add_argument("input_file", type=str, help="Path to the file containing prompts (one per line)")
    parser.add_argument("--model", type=str, default="minilm",
                        help="Embedding model: 'minilm', 'mpnet', or 'distilroberta' (default: minilm)")
    parser.add_argument("--dataset", type=str, default="dataset_secure.json",
                        help="Path to the dataset JSON file (default: dataset_secure.json)")
    parser.add_argument("--output", type=str, default="input_prompt.csv",
                        help="Path to the CSV file to create (default: input_prompt.csv)")
    parser.add_argument("--topk", type=int, default=3,
                        help="Number of candidate vulnerabilities to return (default: 3)")
    args = parser.parse_args()

    # Carica il modello di embedding in base alla scelta dell'utente
    embedding_model = get_embedding_model(args.model)
    prompts, cwe_mapping = load_dataset(args.dataset)
    embeddings = preprocess_prompts(prompts, embedding_model)
    scores = [p["score"] for p in prompts]
    regressor = train_model(embeddings, scores)

    # Leggi i nuovi prompt
    with open(args.input_file, "r") as f:
        new_prompts = [line.strip() for line in f if line.strip()]

    # Crea il file CSV da zero, scrivendo l'header
    with open(args.output, mode="w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        header = [
            "prompt", "predicted_probability",
            "candidate_cwe_1", "probcwe_1",
            "candidate_cwe_2", "probcwe_2",
            "candidate_cwe_3", "probcwe_3"
        ]
        writer.writerow(header)
        
        # Processa ogni prompt
        for prompt in new_prompts:
            predicted_probability, candidates = classify_prompt(
                prompt, embedding_model, embeddings, prompts, regressor, cwe_mapping, top_k=args.topk
            )
            # Prepara i dati per l'output CSV
            row = [prompt, round(predicted_probability, 2)]
            for candidate in candidates:
                candidate_cwe, prob = candidate
                row.extend([
                    candidate_cwe if candidate_cwe is not None else "None",
                    round(prob, 2) if prob is not None else "None"
                ])
            writer.writerow(row)
            print(f"\nPrompt: {prompt}")
            print(f"Predicted Vulnerability Probability: {round(predicted_probability, 2)}")
            print("Candidate Vulnerabilities (CWE, Normalized Probability):")
            for candidate in candidates:
                candidate_cwe, prob = candidate
                print(f"  - {candidate_cwe}: {round(prob, 2) if prob is not None else 'None'}")
