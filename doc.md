#Overview
This project implements a Streamlit-based application that classifies user-provided prompts using a pre-trained language model. It assesses the prompt's similarity with a dataset of existing prompts and assigns a score based on similarity-weighted scoring. The application also checks for security vulnerabilities in the prompt and provides improvement suggestions.
##Dependencies:
streamlit: For UI rendering

json: To handle the dataset

numpy: For numerical operations

matplotlib.pyplot: For plotting similarity distribution

logging: To log operations and user inputs

sentence_transformers: To generate text embeddings and calculate similarity
## Functions
### load_dataset(json_file)
Parameters:

json_file (str): Path to the dataset file in JSON format.

Returns:

prompts (list of dict): A list of prompts, each containing a text (string) and score (float).

Description:
Loads and processes the dataset from the JSON file. The dataset is structured as a dictionary where categories contain prompts with associated scores.


### preprocess_prompts(prompts, model)

Parameters:

prompts (list of dict): List of prompts.

model (SentenceTransformer): The language model used to generate embeddings.

Returns:

embeddings (tensor): Precomputed embeddings for all stored prompts.

Description:
Converts the text of each prompt into numerical embeddings using SentenceTransformer.

### classify_prompt(new_prompt, model, embeddings, prompts)

Parameters:

new_prompt (str): The user-provided prompt.

model (SentenceTransformer): The language model used for embeddings.

embeddings (tensor): Precomputed embeddings of stored prompts.

prompts (list of dict): List of stored prompts.

Returns:

final_score (float): Weighted mean score based on similarity.

best_match_score (float): Highest similarity score.

similarities (array): Similarity scores with all stored prompts.

Description:
Embeds the new prompt, calculates similarity with stored prompts, and computes a weighted mean score.

### analyze_vulnerabilities(prompt)

Parameters:

prompt (str): The user-provided prompt.

Returns:

vulnerabilities (list of str): Identified security risks in the prompt.

Description:
Checks for common security vulnerabilities like eval(), input(), and unsafe file operations.

### improvement_suggestions(score)

Parameters:

score (float): The computed score of the prompt.

Returns:

suggestion (str): Advice on how to improve the prompt.

Description:
Provides recommendations for improving the prompt based on its score.
