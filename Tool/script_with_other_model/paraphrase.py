import json
import numpy as np
import logging
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.ensemble import RandomForestRegressor

# Configura logging
logging.basicConfig(filename='log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

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

# Genera embeddings per i prompt
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
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X, y)
    logging.info("Model trained successfully")
    return regressor

# Classifica un nuovo prompt e trova la CWE corrispondente
def classify_prompt(new_prompt, model, embeddings, prompts, regressor, cwe_mapping):
    logging.info(f"Classifying new prompt: {new_prompt}")
    new_embedding = model.encode(new_prompt, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(new_embedding, embeddings)[0]
    best_match_idx = np.argmax(similarities.cpu().numpy())
    best_match_score = similarities[best_match_idx].cpu().item()
    predicted_score = regressor.predict([new_embedding.cpu().numpy()])[0]
    best_match_text = prompts[best_match_idx]["text"]
    best_match_cwe = cwe_mapping.get(best_match_text, "Unknown CWE")
    # Troncamento della CWE: rimuove tutto ciò che segue (incluso) il carattere "_"
    best_match_cwe = best_match_cwe.split('_')[0]
    logging.info(f"Best match similarity: {best_match_score}")
    logging.info(f"Predicted score: {predicted_score}")
    logging.info(f"Associated CWE (troncata): {best_match_cwe}")
    return predicted_score, best_match_score, best_match_cwe

# Esegui il codice da terminale
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Classify prompts and find CWE vulnerabilities using paraphrase-mpnet-base-v2.")
    parser.add_argument("input_file", type=str, help="Path to the file containing prompts (one per line)")
    args = parser.parse_args()

    # Utilizza il modello paraphrase-mpnet-base-v2
    model = SentenceTransformer("paraphrase-mpnet-base-v2")
    json_file = "dataset_secure.json"
    prompts, cwe_mapping = load_dataset(json_file)
    embeddings = preprocess_prompts(prompts, model)
    scores = [p["score"] for p in prompts]
    regressor = train_model(embeddings, scores)

    with open(args.input_file, "r") as f:
        new_prompts = [line.strip() for line in f if line.strip()]

    for prompt in new_prompts:
        score, similarity, cwe = classify_prompt(prompt, model, embeddings, prompts, regressor, cwe_mapping)
        print(f"\nPrompt: {prompt}")
        print(f"Predicted Score: {round(score, 2)}")
        print(f"Best Match Similarity: {round(similarity, 2)}")
        print(f"Associated CWE: {cwe}")
