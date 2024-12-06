import pandas as pd
import asyncio
import os
import re
import logging
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)

def get_llm_service_function(model_name: str):
    import importlib
    module_name = f'llm_services.get_{model_name}_response'
    function_name = f'get_{model_name}_response'
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
        return func
    except (ImportError, AttributeError) as e:
        raise ImportError(f"LLM service function for model '{model_name}' not found") from e

async def add_judgments(
    judgments_file_path: str,
    judgment_model: str,
    version_name: str
):
    # Verify that the file exists
    if not os.path.exists(judgments_file_path):
        raise FileNotFoundError(f"The file {judgments_file_path} does not exist.")

    # Read JSONL file
    try:
        with open(judgments_file_path, 'r') as f:
            if os.stat(judgments_file_path).st_size == 0:
                raise ValueError("The input file is empty.")
            df = pd.read_json(f, lines=True)
    except ValueError as e:
        logging.error(f"Error reading JSON file: {e}")
        return

    v1_model = df['v1_model'].iloc[0]
    v2_model = df['v2_model'].iloc[0]
    logging.info(f"Models: v1_model = {v1_model}, v2_model = {v2_model}")

    # Get the function for the judgment model
    llm_service_function = get_llm_service_function(judgment_model)

    output_file = f"./Data/Output/judgments/{version_name}_big_c_test_{v1_model}_vs_{v2_model}.jsonl"
    # check if the file exists
    if os.path.exists(output_file):
        logging.info(f"File {output_file} already exists. Skipping.")
        return
    logging.info(f"Output file: {output_file}")

    for index, row in df.iterrows():
        try:
            logging.info(f"Processing row {index}")
            response = await llm_service_function(row['full_judgment_prompt'])
            response = int(response)
            logging.info(f"Received response for row {index}: {response}")
            try:
                winner = ''
                if response == 1:
                    winner = v1_model
                elif response == 2:
                    winner = v2_model
                elif response == 3:
                    winner = 'tie'
                else:
                    winner = 'unknown'
                df.at[index, f'{judgment_model}_judgment'] = winner
            except ValueError as e:
                logging.error(f"Error converting response to int for row {index}: {e}")
                df.at[index, f'{judgment_model}_judgment'] = response
        except Exception as e:
            logging.error(f"Error processing row {index}: {e}")

    if not df.empty:
        df.to_json(output_file, orient='records', lines=True)
        logging.info(f"Results saved to {output_file}")
    else:
        logging.warning("No data to save.")

async def main():
    await add_judgments(
        judgments_file_path="./Data/Output/judgments/v0_big_c_test_google_translate_vs_sonnet_3_point_5.jsonl",
        judgment_model="gpt_4o",
        version_name="v1"
    )

# add judgments for all judgment files
# async def main():
#     files = os.listdir("./Data/Output/judgments/")
#     for file in files:
#         await get_judgments(
#             judgments_file_path=f"./Data/Output/judgments/{file}",
#             judgment_model="gpt_4o"
#         )

if __name__ == "__main__":
    asyncio.run(main())