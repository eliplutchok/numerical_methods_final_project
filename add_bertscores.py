import pandas as pd
import os
import logging
from bert_score import score
import torch

logging.basicConfig(level=logging.INFO)

def add_bertscores(
    model_name: str,
    file_path: str,
    version_name: str,
    lang: str = 'en'
):
    # Verify that the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Read JSONL file
    try:
        with open(file_path, 'r') as f:
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

    # Prepare data for BERTScore
    refs = df['joined_english_sentences'].tolist()
    cands = df[f'{model_name}_translation'].tolist()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Compute BERTScore for translations
    _, _, F1 = score(cands, refs, lang=lang, verbose=True, device=device)

    # Add BERTScore to DataFrame
    df[f'{model_name}_bertscore'] = F1.tolist()

    if not df.empty:
        df.to_json(output_file, orient='records', lines=True)
        logging.info(f"Results saved to {output_file}")
    else:
        logging.warning("No data to save.")

    return output_file

def main():
    model_name = "sonnet_3_point_5"
    file_path = "./Data/Output/translations/big_c_conversations_test_sonnet_3_point_5.jsonl"
    version_name = "v2"
    output_file = add_bertscores(model_name, file_path, version_name)
    print(output_file)

if __name__ == "__main__":
    main()
