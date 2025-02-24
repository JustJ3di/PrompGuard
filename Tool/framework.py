import json
import numpy as np
import logging
from sentence_transformers import SentenceTransformer, util

# Configura logging
logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load dataset
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

# Preprocess prompts
def preprocess_prompts(prompts, model):
    logging.info("Generating embeddings for stored prompts")
    embeddings = model.encode([p["text"] for p in prompts], convert_to_tensor=True)
    logging.info("Embeddings generated successfully")
    return embeddings

# Classify prompt and analyze vulnerabilities
def classify_prompt(new_prompt, model, embeddings, prompts):
    logging.info(f"Classifying new prompt: {new_prompt}")
    new_embedding = model.encode(new_prompt, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(new_embedding, embeddings)[0]
    best_match_idx = np.argmax(similarities.cpu().numpy())
    best_match_score = similarities[best_match_idx].cpu().item()
    
    # Weighted scoring
    scores = np.array([p["score"] for p in prompts])
    #weighted_scores = scores * similarities.cpu().numpy()
    #final_score = sum(weighted_scores) / sum(similarities.cpu().numpy()) #weighted mean for the score
    tau = 0.5  # Ignora le similarità sotto questa soglia
    filtered_indices = np.where(similarities.cpu().numpy() > tau)[0]
    if len(filtered_indices) > 0:
        filtered_scores = np.array([scores[i] for i in filtered_indices])
        filtered_similarities = np.array([similarities[i].cpu().numpy() for i in filtered_indices])
        weighted_scores = filtered_scores * filtered_similarities
        final_score = sum(weighted_scores) / sum(filtered_similarities)
    else:
        final_score = np.mean(scores)  # Default nel caso in cui nessun prompt superi la soglia
    
    logging.info(f"Best match similarity: {best_match_score}")
    logging.info(f"Final calculated score: {final_score}")
    
    return final_score, best_match_score, similarities.cpu().numpy()

# Analyze vulnerabilities
def analyze_vulnerabilities(prompt):
    logging.info(f"Analyzing potential vulnerabilities in prompt: {prompt}")
    vulnerabilities = []
    if "input(" in prompt or "eval(" in prompt:
        vulnerabilities.append("Potential injection vulnerability: Avoid direct user input evaluation.")
    if "open(" in prompt and "w" in prompt:
        vulnerabilities.append("File handling risk: Ensure proper access control and validation.")
    return vulnerabilities

# Main execution block
def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    json_file = "dataset_secure.json"
    logging.info("Initializing model and loading dataset")
    prompts = load_dataset(json_file)
    embeddings = preprocess_prompts(prompts, model)
    logging.info("Model and dataset ready")

    while True:
        new_prompt = input("Enter your prompt (or type 'exit' to quit): ")
        if new_prompt.lower() == 'exit':
            break
        
        try:
            logging.info(f"Processing new prompt classification: {new_prompt}")
            score, similarity, _ = classify_prompt(new_prompt, model, embeddings, prompts)
            vulnerabilities = analyze_vulnerabilities(new_prompt)
            
            print(f"\nAssigned Score: {round(score, 2)}")
            print(f"Similarity: {similarity:.2f}")
            
            if vulnerabilities:
                print("\n⚠️ Potential Vulnerabilities Found:")
                for vuln in vulnerabilities:
                    print(f"- {vuln}")
            
            logging.info("Classification process completed successfully")
        
        except ValueError as e:
            logging.error(f"Error during classification: {e}")
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
