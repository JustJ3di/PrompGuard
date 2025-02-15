import streamlit as st
import json
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

# Load dataset
def load_dataset(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)
    prompts = []
    for category, items in data.items():
        for text, score in items.items():
            prompts.append({"text": text, "score": score})
    return prompts

# Preprocess prompts
def preprocess_prompts(prompts, model):
    embeddings = model.encode([p["text"] for p in prompts], convert_to_tensor=True)
    return embeddings

# Classify prompt and analyze vulnerabilities
def classify_prompt(new_prompt, model, embeddings, prompts):
    new_embedding = model.encode(new_prompt, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(new_embedding, embeddings)[0]
    best_match_idx = np.argmax(similarities.cpu().numpy())
    best_match_score = similarities[best_match_idx].cpu().item()
    
    # Weighted scoring
    scores = np.array([p["score"] for p in prompts])
    weighted_scores = scores * similarities.cpu().numpy()
    final_score = sum(weighted_scores) / sum(similarities.cpu().numpy())
    
    return final_score, best_match_score, similarities.cpu().numpy()

# Analyze vulnerabilities
def analyze_vulnerabilities(prompt):
    vulnerabilities = []
    if "input(" in prompt or "eval(" in prompt:
        vulnerabilities.append("Potential injection vulnerability: Avoid direct user input evaluation.")
    if "open(" in prompt and "w" in prompt:
        vulnerabilities.append("File handling risk: Ensure proper access control and validation.")
    return vulnerabilities

# Provide improvement suggestions
def improvement_suggestions(score):
    if score > 2.5:
        return "Consider refactoring the prompt to include security best practices."
    elif score > 1.5:
        return "The prompt is decent but could be improved with stricter validation measures."
    else:
        return "The prompt is well-structured with good security considerations."

# UI with Streamlit
st.title("Prompt Classification with LLM")
st.write("Enter a prompt to get the assigned score, similarity, and security suggestions.")

model = SentenceTransformer("all-MiniLM-L6-v2")
json_file = "dataset.json"
prompts = load_dataset(json_file)
embeddings = preprocess_prompts(prompts, model)

new_prompt = st.text_area("Enter your prompt:")
if st.button("Classify"):
    if new_prompt:
        try:
            score, similarity, similarities = classify_prompt(new_prompt, model, embeddings, prompts)
            vulnerabilities = analyze_vulnerabilities(new_prompt)
            suggestion = improvement_suggestions(score)
            
            st.metric(label="Assigned Score", value=round(score, 2))
            st.metric(label="Similarity", value=f"{similarity:.2f}")
            
            if vulnerabilities:
                st.warning("⚠️ Potential Vulnerabilities Found:")
                for vuln in vulnerabilities:
                    st.write(f"- {vuln}")
            
            st.success(f"💡 Suggestion: {suggestion}")
            
            # Plot similarity distribution
            fig, ax = plt.subplots()
            ax.hist(similarities, bins=10, color='skyblue', edgecolor='black')
            ax.set_title("Similarity Distribution")
            ax.set_xlabel("Similarity Score")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
        
        except ValueError as e:
            st.error(str(e))
    else:
        st.warning("Please enter a valid prompt.")