import json
import numpy as np
import logging
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Configura logging
logging.basicConfig(filename='log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Carica il modello di embedding (Sentence-Transformers)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Carica il modello di LLM open-source per generare testo (modello leggero)
llm_pipeline = pipeline("text-generation", model="distilgpt2")

# Funzione per generare una chat completion (esempio: un haiku su AI)
def get_chat_completion(prompt):
    response = llm_pipeline(prompt, max_length=50, do_sample=True, temperature=0.7)
    return response[0]["generated_text"].strip()

# Esempio: genera un haiku su AI e lo stampa
haiku = get_chat_completion("Write a haiku about AI")
print("Generated Haiku:")
print(haiku)
print("=" * 50)

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

# Funzione per ottenere embeddings con Sentence-Transformers
def get_embedding(text):
    return embedding_model.encode(text, convert_to_numpy=True)

# Genera embeddings per i prompt (in batch)
def preprocess_prompts(prompts):
    logging.info("Generating embeddings for stored prompts using Sentence-Transformers")
    texts = [p["text"] for p in prompts]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    logging.info("Embeddings generated successfully")
    return embeddings

# Addestra il modello di regressione
def train_model(embeddings, scores):
    logging.info("Training Random Forest model on embeddings")
    X = embeddings
    y = np.array(scores)
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X, y)
    logging.info("Model trained successfully")
    return regressor

# Classifica un nuovo prompt e trova la CWE corrispondente
def classify_prompt(new_prompt, prompts, embeddings, regressor, cwe_mapping):
    logging.info(f"Classifying new prompt: {new_prompt}")
    new_embedding = get_embedding(new_prompt)
    # Calcola la similarità coseno tra il nuovo embedding e quelli esistenti
    similarities = cosine_similarity(new_embedding.reshape(1, -1), embeddings)[0]
    best_match_idx = np.argmax(similarities)
    best_match_score = similarities[best_match_idx]
    predicted_score = regressor.predict(new_embedding.reshape(1, -1))[0]
    best_match_text = prompts[best_match_idx]["text"]
    best_match_cwe = cwe_mapping.get(best_match_text, "Unknown CWE")
    # Troncamento della CWE: rimuove tutto ciò che segue (incluso) il carattere "_"
    best_match_cwe = best_match_cwe.split('_')[0]
    
    logging.info(f"Best match similarity: {best_match_score}")
    logging.info(f"Predicted score: {predicted_score}")
    logging.info(f"Associated CWE (troncata): {best_match_cwe}")
    
    return predicted_score, best_match_score, best_match_cwe

# Esegui il codice da terminale per classificare nuovi prompt
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Classify prompts and find CWE vulnerabilities using Sentence-Transformers and a light LLM (distilgpt2).")
    parser.add_argument("input_file", type=str, help="Path to the file containing prompts (one per line)")
    args = parser.parse_args()
  
    json_file = "dataset_secure.json"
    prompts, cwe_mapping = load_dataset(json_file)
    embeddings = preprocess_prompts(prompts)
    scores = [p["score"] for p in prompts]
    regressor = train_model(embeddings, scores)

    with open(args.input_file, "r") as f:
        new_prompts = [line.strip() for line in f if line.strip()]

    for prompt in new_prompts:
        score, similarity, cwe = classify_prompt(prompt, prompts, embeddings, regressor, cwe_mapping)
        print(f"\nPrompt: {prompt}")
        print(f"Predicted Score: {round(score, 2)}")
        print(f"Best Match Similarity: {round(similarity, 2)}")
        print(f"Associated CWE: {cwe}")
