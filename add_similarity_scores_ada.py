import pandas as pd
import os
import logging
import openai
import numpy as np
from typing import Optional
from langchain.embeddings.openai import OpenAIEmbeddings

logging.basicConfig(level=logging.INFO)


def add_similarity_scores(
    file_path: str,
    version_name: str,
    embedding_model: str = 'text-embedding-ada-002',
    translation_column: Optional[str] = None,
    lang: str = 'en'
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
    output_filename = f"{version_name}_{input_filename}"
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

    # Prepare data
    refs = df['joined_english_sentences'].tolist()

    # Detect translation column if not provided
    if translation_column is None:
        translation_columns = [col for col in df.columns if col.endswith("_translation")]
        if len(translation_columns) == 0:
            raise ValueError("No translation column found in the DataFrame.")
        elif len(translation_columns) > 1:
            raise ValueError(f"Multiple translation columns found: {translation_columns}. Please specify the translation column.")
        else:
            translation_column = translation_columns[0]
            logging.info(f"Using translation column: {translation_column}")

    cands = df[translation_column].tolist()

    # Initialize embeddings
    embed = OpenAIEmbeddings(
        model=embedding_model,
        openai_api_key=openai.api_key
    )

    # Compute embeddings
    logging.info("Computing embeddings for references...")
    refs_embeddings = embed.embed_documents(refs)
    logging.info("Computing embeddings for translations...")
    cands_embeddings = embed.embed_documents(cands)

    # Compute similarity scores
    similarity_scores = []
    for i in range(len(refs_embeddings)):
        v1 = np.array(refs_embeddings[i])
        v2 = np.array(cands_embeddings[i])
        similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        similarity_scores.append(similarity)

    # Add similarity score to DataFrame
    # Extract the translation model name from translation_column
    # e.g., if translation_column is 'aya_8b_translation', translation_model_name is 'aya_8b'
    translation_model_name = translation_column.replace('_translation', '')
    similarity_column_name = f"{translation_model_name}_{embedding_model}_similarity"
    df[similarity_column_name] = similarity_scores
 
    if not df.empty:
        df.to_json(output_file, orient='records', lines=True, force_ascii=False)
        logging.info(f"Results saved to {output_file}")
    else:
        logging.warning("No data to save.")

def main():
    file_path = "./Data/Output/translations/v2_big_c_conversations_test_sonnet_3_point_5.jsonl"
    version_name = "a"
    embedding_model = 'text-embedding-ada-002'
    # Optionally specify translation column
    translation_column = None  # or set to the specific column name, e.g., 'aya_8b_translation'

    add_similarity_scores(file_path, version_name, embedding_model, translation_column)

if __name__ == "__main__":
    main()
