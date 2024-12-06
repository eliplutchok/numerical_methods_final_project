import pandas as pd
import os
import logging
import openai
import numpy as np
from typing import Optional
from langchain.embeddings.openai import OpenAIEmbeddings

logging.basicConfig(level=logging.INFO)

def compute_t1_t2_similarity(
    file_path: str,
    version_name: str,
    embedding_model: str = 'text-embedding-ada-002',
    models: Optional[list] = None
):
    # Verify that the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Read JSONL file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if os.stat(file_path).st_size == 0:
                raise ValueError("The input file is empty.")
            df = pd.read_json(f, lines=True)
    except ValueError as e:
        logging.error(f"Error reading JSON file: {e}")
        return

    # Construct the output file path with version_name as prefix to filename
    input_dir, input_filename = os.path.split(file_path)

    # Remove the old version number from the beginning of the filename
    filename_parts = input_filename.split('_')
    # Check if the first part is a version number (e.g., 'v1', 'v2')
    if filename_parts[0].startswith('v') and filename_parts[0][1:].isdigit():
        # Remove the old version number
        input_filename_no_version = '_'.join(filename_parts[1:])
    else:
        input_filename_no_version = input_filename  # No version prefix found

    # Create the new output filename with your new version number
    output_filename = f"{version_name}_{input_filename_no_version}"
    output_file = os.path.join(input_dir, output_filename)

    # Check if the output file already exists
    if os.path.exists(output_file):
        logging.info(f"File {output_file} already exists. Skipping.")
        return

    logging.info(f"Processing file: {file_path}")
    logging.info(f"Output file will be: {output_file}")

    # Set your OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OpenAI API key is not set in the environment variable 'OPENAI_API_KEY'.")

    # If models are not specified, infer them from the DataFrame columns
    if models is None:
        translation_columns = [col for col in df.columns if '_translation_t1' in col]
        models = [col.replace('_translation_t1', '') for col in translation_columns]
        logging.info(f"Models inferred: {models}")

    # Initialize embeddings
    embed = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_key=openai.api_key
    )

    # Dictionary to hold similarity scores for each model
    similarity_scores_dict = {}

    # Compute similarity scores for each model
    for model in models:
        t1_column = f"{model}_translation_t1"
        t2_column = f"{model}_translation_t2"
        similarity_column_name = f"{model}_t1_t2_{embedding_model}_similarity"

        # Check if both columns exist
        if t1_column not in df.columns or t2_column not in df.columns:
            logging.warning(f"Translation columns for model '{model}' not found. Skipping.")
            continue

        t1_translations = df[t1_column].tolist()
        t2_translations = df[t2_column].tolist()

        # add warning if there are any missing values
        if df[t1_column].isnull().any() or df[t2_column].isnull().any():
            logging.warning(f"Missing values detected in {t1_column} or {t2_column}. Skipping.")
            # continue

        # Clean translations to ensure all elements are strings and handle missing values
        t1_translations = [str(x) if pd.notnull(x) else "" for x in t1_translations]
        t2_translations = [str(x) if pd.notnull(x) else "" for x in t2_translations]

        # Check for non-string entries (optional)
        non_string_t1 = [x for x in t1_translations if not isinstance(x, str)]
        non_string_t2 = [x for x in t2_translations if not isinstance(x, str)]
        if non_string_t1:
            logging.warning(f"Non-string values detected in {t1_column}: {non_string_t1}")
        if non_string_t2:
            logging.warning(f"Non-string values detected in {t2_column}: {non_string_t2}")

        # Compute embeddings
        logging.info(f"Computing embeddings for model '{model}' translations (t1)...")
        t1_embeddings = embed.embed_documents(t1_translations)
        logging.info(f"Computing embeddings for model '{model}' translations (t2)...")
        t2_embeddings = embed.embed_documents(t2_translations)

        # Compute similarity scores
        similarity_scores = []
        for i in range(len(t1_embeddings)):
            v1 = np.array(t1_embeddings[i])
            v2 = np.array(t2_embeddings[i])
            similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            similarity_scores.append(similarity)

        # Add similarity score to DataFrame
        df[similarity_column_name] = similarity_scores
        similarity_scores_dict[model] = similarity_scores

    # Determine the model with the higher similarity score for each instance
    if similarity_scores_dict:
        models_list = list(similarity_scores_dict.keys())
        num_instances = len(df)
        consistency_judgments = []

        for i in range(num_instances):
            max_similarity = -1
            best_model = None
            for model in models_list:
                similarity = similarity_scores_dict[model][i]
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_model = model
            consistency_judgments.append(best_model)

        # Add the consistency judgment column
        df[f'{embedding_model}_consistency_judgment'] = consistency_judgments

    if not df.empty:
        df.to_json(output_file, orient='records', lines=True, force_ascii=False)
        logging.info(f"Results saved to {output_file}")
    else:
        logging.warning("No data to save.")

def main():
    folder_path = "./Data/Output/consistency_scores/"
    version_name = "v1"
    embedding_model = 'text-embedding-ada-002'

    files = os.listdir(folder_path)
    for file in files:
        if not file.startswith('v0_'):
            continue
        if not file.endswith('.jsonl'):
            continue
        print(file)
        file_path = os.path.join(folder_path, file)
        compute_t1_t2_similarity(file_path, version_name, embedding_model)

if __name__ == "__main__":
    main()