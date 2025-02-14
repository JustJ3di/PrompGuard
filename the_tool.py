import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
import datetime

def log_operation(operation):
    with open("log.txt", "a") as log_file:
        log_file.write(f"{datetime.datetime.now()} - {operation}\n")

def load_dataset(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)
    prompts = []
    for category, items in data.items():
        for text, score in items.items():
            prompts.append({"text": text, "score": score})
    return prompts

def preprocess_prompts(prompts, model):
    embeddings = model.encode([p["text"] for p in prompts], convert_to_tensor=True)
    return embeddings

def classify_prompt(new_prompt, model, embeddings, prompts):
    new_embedding = model.encode(new_prompt, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(new_embedding, embeddings)[0]
    similarities = similarities.cpu().numpy()
    
    # Ponderazione dei punteggi basata sulla similarità
    weighted_scores = np.array([p["score"] for p in prompts]) * similarities
    final_score = np.sum(weighted_scores) / np.sum(similarities)
    
    best_match_idx = np.argmax(similarities)
    best_match_score = similarities[best_match_idx]
    
    log_operation(f"Processed prompt: {new_prompt} | Weighted Score: {final_score} | Best Match Similarity: {best_match_score}")
    return final_score, best_match_score

# UI with Streamlit
st.title("Prompt Classification with LLM")
st.write("Enter a prompt to get the assigned weighted score and similarity compared to existing data.")

model = SentenceTransformer("all-MiniLM-L6-v2")
json_file = "dataset.json"
prompts = load_dataset(json_file)
embeddings = preprocess_prompts(prompts, model)

new_prompt = st.text_area("Enter your prompt:")
if st.button("Classify"):
    if new_prompt:
        try:
            score, similarity = classify_prompt(new_prompt, model, embeddings, prompts)
            st.metric(label="Weighted Score", value=round(score, 2))
            st.metric(label="Best Match Similarity", value=f"{similarity:.2f}")
        except ValueError as e:
            st.error(str(e))
    else:
        st.warning("Please enter a valid prompt.")


