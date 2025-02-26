import json
import numpy as np
import logging
from sentence_transformers import SentenceTransformer, util
from sklearn.ensemble import RandomForestRegressor
import torch

# Configura logging
logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Carica dataset
def load_dataset(json_file):
    logging.info(f"Loading dataset from {json_file}")
    with open(json_file, "r") as file:
        data = json.load(file)
    prompts = []
    for category, items in data.items():
        for text, score in items.items():
            prompts.append({"text": text, "score": score})
    logging.info(f"Loaded {len(prompts)} prompts from dataset")
    return prompts

# Genera embeddings
def preprocess_prompts(prompts, model):
    logging.info("Generating embeddings for stored prompts")
    embeddings = model.encode([p["text"] for p in prompts], convert_to_tensor=True)
    logging.info("Embeddings generated successfully")
    return embeddings

# Addestra il modello di regressione
def train_model(embeddings, scores):
    logging.info("Training Random Forest model on embeddings")
    X = embeddings.cpu().numpy()
    y = np.array(scores)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    logging.info("Model trained successfully")
    return model

# Classifica un nuovo prompt
def classify_prompt(new_prompt, model, embeddings, prompts, regressor):
    logging.info(f"Classifying new prompt: {new_prompt}")
    
    # Genera embedding del nuovo prompt
    new_embedding = model.encode(new_prompt, convert_to_tensor=True)
    
    # Calcola similarità con i prompt esistenti
    similarities = util.pytorch_cos_sim(new_embedding, embeddings)[0]
    best_match_idx = np.argmax(similarities.cpu().numpy())
    best_match_score = similarities[best_match_idx].cpu().item()

    # Predice lo score con il modello pre-addestrato
    predicted_score = regressor.predict([new_embedding.cpu().numpy()])[0]
    
    logging.info(f"Best match similarity: {best_match_score}")
    logging.info(f"Predicted score: {predicted_score}")

    return predicted_score, best_match_score

# Esegui il codice da terminale
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify prompts from a file using a trained model.")
    parser.add_argument("input_file", type=str, help="Path to the file containing prompts (one per line)")
    args = parser.parse_args()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    json_file = "dataset_secure.json"

    # Carica dataset e genera embeddings
    prompts = load_dataset(json_file)
    embeddings = preprocess_prompts(prompts, model)

    # Addestra il modello di regressione
    scores = [p["score"] for p in prompts]
    regressor = train_model(embeddings, scores)

    # Legge i prompt dal file
    with open(args.input_file, "r") as f:
        new_prompts = [line.strip() for line in f if line.strip()]

    # Classifica ogni nuovo prompt
    for prompt in new_prompts:
        score, similarity = classify_prompt(prompt, model, embeddings, prompts, regressor)
        print(f"\nPrompt: {prompt}")
        print(f"Predicted Score: {round(score, 2)}")
        print(f"Best Match Similarity: {round(similarity, 2)}\n")
