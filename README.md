# best_prompt
## to do
### 1) Formare un dataset comodo :white_check_mark:
Creare nuovo dataset json con
{ "CWE":{"prompt_type1":punteggio1,"prompt_type2":punteggio2...},"CWE":{"prompt_type1":punteggio1,...},...ecc} 
### 2) Allenare modello preesistente tramite Ollama/hugging_face  :white_check_mark:
Allenare un modello tale per cui dato un prompt mi dice il punteggio associato, quindi allenarlo secondo teniche di verosomiglianza testuale.
### 3) Creare un tool  :white_check_mark:
Creare un tool che offre la possibilità ad un utente finale, dato un prompt di avere un punteggio relativo a quanta probabilità a questo di generargli del codice funzionante e sicuro.

HOW THE TOOL WORK

Prompt Classification - Scoring Details

How the Score is Assigned

1. Calculating Similarities

The SentenceTransformer model encodes the new prompt into a vector embedding.

The SentenceTransformer model encodes the new prompt into a vector embedding.

The result is an array of similarity scores, where each value represents how similar the new prompt is to those in the dataset.

2. Weighting the Scores

Each existing prompt in the dataset has a predefined score.

The scores are multiplied by their corresponding similarity values, generating a weighted score array:
#### weighted_scores = scores * similarities



3. Final Score Calculation

The final score is obtained by summing the weighted scores and normalizing them by the sum of similarities:

#### final_score = sum(scores * similarities) / sum(similarities)

This ensures that prompts with higher similarity have a greater influence on the final score.

4. Best Match Selection

The prompt with the highest similarity is identified, and its similarity value is reported as the "Best Match Similarity."

5. Logging the Operation

Every operation is recorded in log.txt, including:

The input prompt

The assigned weighted score

The highest similarity match

This method provides a dynamic and adaptive scoring system that considers how closely a new prompt aligns with existing ones in the dataset.
