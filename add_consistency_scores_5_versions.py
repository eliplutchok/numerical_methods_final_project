import pandas as pd
import os
import logging
import openai
import numpy as np
from itertools import combinations
from typing import Optional
from langchain.embeddings.openai import OpenAIEmbeddings

logging.basicConfig(level=logging.INFO)

def compute_consistency_across_versions(
    file_path: str,
    version_name: str,
    embedding_model: str = 'text-embedding-ada-002',
    models: Optional[list] = None,
    translation_versions: list = None
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

    # If translation_versions are not specified, default to ['t1', 't2', 't3', 't4', 't5']
    if translation_versions is None:
        translation_versions = ['t1', 't2', 't3', 't4', 't5']

    # Initialize embeddings
    embed = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_key=openai.api_key
    )

    # Dictionary to hold consistency scores for each model
    consistency_scores_dict = {}

    # Compute consistency scores for each model
    for model in models:
        translation_columns = [f"{model}_translation_{t}" for t in translation_versions]

        # Check if all translation columns exist
        missing_columns = [col for col in translation_columns if col not in df.columns]
        if missing_columns:
            logging.warning(f"Translation columns missing for model '{model}': {missing_columns}. Skipping.")
            continue

        # Gather translations
        translations = {}
        for t in translation_versions:
            col_name = f"{model}_translation_{t}"
            translations[t] = df[col_name].tolist()

        # Check for missing values and handle them
        for t in translation_versions:
            translations[t] = [str(x) if pd.notnull(x) else "" for x in translations[t]]

        # Compute embeddings for each translation version
        embeddings = {}
        for t in translation_versions:
            logging.info(f"Computing embeddings for model '{model}' translations ({t})...")
            embeddings[t] = embed.embed_documents(translations[t])

        # Compute pairwise similarities
        pairwise_similarities = []
        version_pairs = list(combinations(translation_versions, 2))
        num_instances = len(df)
        for idx in range(num_instances):
            similarities = []
            for (t1, t2) in version_pairs:
                v1 = np.array(embeddings[t1][idx])
                v2 = np.array(embeddings[t2][idx])
                similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                similarities.append(similarity)
            # Average similarity for this instance
            average_similarity = np.mean(similarities)
            pairwise_similarities.append(average_similarity)

        # Add consistency score to DataFrame
        consistency_column_name = f"{model}_{embedding_model}_consistency_score"
        df[consistency_column_name] = pairwise_similarities
        consistency_scores_dict[model] = pairwise_similarities

    # Determine the model with the higher consistency score for each instance
    if consistency_scores_dict:
        models_list = list(consistency_scores_dict.keys())
        num_instances = len(df)
        consistency_judgments = []

        for i in range(num_instances):
            max_consistency = -1
            best_model = None
            for model in models_list:
                consistency = consistency_scores_dict[model][i]
                if consistency > max_consistency:
                    max_consistency = consistency
                    best_model = model
            consistency_judgments.append(best_model)

        # Add the consistency judgment column
        df[f'{embedding_model}_consistency_judgment_5_versions'] = consistency_judgments

    if not df.empty:
        df.to_json(output_file, orient='records', lines=True, force_ascii=False)
        logging.info(f"Results saved to {output_file}")
    else:
        logging.warning("No data to save.")

def main():
    folder_path = "./Data/Output/consistency_scores_5_versions/"
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
        compute_consistency_across_versions(file_path, version_name, embedding_model)

if __name__ == "__main__":
    main()