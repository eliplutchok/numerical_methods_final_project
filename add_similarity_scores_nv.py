import pandas as pd
import os
import logging
import numpy as np
import requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)

def get_embedding(input_text, model, api_key, input_type="query", encoding_format="float", truncate="NONE"):
    url = "https://integrate.api.nvidia.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "input": [input_text],
        "model": model,
        "input_type": input_type,
        "encoding_format": encoding_format,
        "truncate": truncate
    }
    logging.info(f"Sending request to NVIDIA API for input: {input_text}")
    try:
        response = requests.post(url, headers=headers, json=data)
        logging.info(f"Received response with status code: {response.status_code}")
        response.raise_for_status()
        logging.info("Response content:")
        logging.info(response.text)
        result = response.json()
        embedding = result['data'][0]['embedding']
        logging.info("Successfully received embedding.")
        return embedding
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
        logging.error(f"Response status code: {response.status_code}")
        logging.error(f"Response text: {response.text}")
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request error occurred: {req_err}")
    except ValueError as json_err:
        logging.error(f"JSON decode error: {json_err}")
        logging.error(f"Response text: {response.text}")
    except KeyError as key_err:
        logging.error(f"Key error: {key_err}")
        try:
            logging.error(f"Response JSON: {response.json()}")
        except Exception as e:
            logging.error(f"Failed to parse response as JSON: {e}")
            logging.error(f"Raw response text: {response.text}")
    return None

def add_similarity_scores(
    file_path: str,
    version_name: str,
    embedding_model: str = 'nvidia/nv-embedqa-mistral-7b-v2',
    translation_column: Optional[str] = None,
    lang: str = 'en'
):
    # Verify that the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Read JSONL file
    try:
        logging.info(f"Reading input file: {file_path}")
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

    # Set your API key
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("API key is not set in the environment variable 'NVIDIA_API_KEY'.")

    # Prepare data
    refs = df['joined_english_sentences'].tolist()
    logging.info(f"Number of reference sentences: {len(refs)}")

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
    logging.info(f"Number of candidate sentences: {len(cands)}")

    # Compute embeddings for references
    logging.info("Computing embeddings for references...")
    refs_embeddings = []
    for idx, ref_text in enumerate(refs):
        embedding = get_embedding(ref_text, embedding_model, api_key)
        if embedding is not None:
            refs_embeddings.append(embedding)
        else:
            logging.error(f"Failed to get embedding for reference at index {idx}.")
            refs_embeddings.append([0]*4096)  # Assuming embedding size is 4096
        if idx % 100 == 0:
            logging.info(f"Computed embeddings for {idx+1}/{len(refs)} references.")

    # Compute embeddings for translations
    logging.info("Computing embeddings for translations...")
    cands_embeddings = []
    for idx, cand_text in enumerate(cands):
        embedding = get_embedding(cand_text, embedding_model, api_key)
        if embedding is not None:
            cands_embeddings.append(embedding)
        else:
            logging.error(f"Failed to get embedding for candidate at index {idx}.")
            cands_embeddings.append([0]*4096)
        if idx % 100 == 0:
            logging.info(f"Computed embeddings for {idx+1}/{len(cands)} candidates.")

    if len(refs_embeddings) != len(cands_embeddings):
        logging.error("The number of reference embeddings does not match the number of candidate embeddings.")
        return

    # Compute similarity scores
    logging.info("Computing similarity scores...")
    similarity_scores = []
    for idx, (ref_embedding, cand_embedding) in enumerate(zip(refs_embeddings, cands_embeddings)):
        v1 = np.array(ref_embedding)
        v2 = np.array(cand_embedding)
        # Handle cases where embeddings might be zeros
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            similarity = 0
        else:
            similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        similarity_scores.append(similarity)
        if idx % 100 == 0:
            logging.info(f"Processed {idx+1}/{len(refs_embeddings)} similarity scores.")

    # Add similarity score to DataFrame
    translation_model_name = translation_column.replace('_translation', '')
    similarity_column_name = f"{translation_model_name}_{embedding_model}_similarity"
    df[similarity_column_name] = similarity_scores

    if not df.empty:
        logging.info(f"Saving results to {output_file}")
        df.to_json(output_file, orient='records', lines=True, force_ascii=False)
        logging.info("Results saved successfully.")
    else:
        logging.warning("No data to save.")

def main():
    # a_v2_big_c_conversations_test_o1_preview
    file_path = "./Data/Output/translations/a_v2_big_c_conversations_test_o1_preview.jsonl"
    version_name = "a"
    embedding_model = 'nvidia/nv-embedqa-mistral-7b-v2'
    # Optionally specify translation column
    translation_column = None  # or set to the specific column name, e.g., 'aya_8b_translation'

    add_similarity_scores(file_path, version_name, embedding_model, translation_column)

if __name__ == "__main__":
    main()
