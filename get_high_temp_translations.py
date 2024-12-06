import asyncio
from typing import List
import pandas as pd
import json
from pathlib import Path
from llm_services.get_gpt_4o_response import get_gpt_4o_response
from llm_services.get_sonnet_3_point_5_response import get_sonnet_3_point_5_response
from llm_services.get_o1_preview_response import get_o1_preview_response
from llm_services.get_o1_mini_response import get_o1_mini_response
from llm_services.get_aya_8b_response import get_aya_8b_response
from llm_services.get_aya_32b_response import get_aya_32b_response
from llm_services.get_llama_3_1_400b_response import get_llama_3_1_400b_response
from llm_services.get_google_translate_response import get_google_translate_response
from llm_services.get_gemini_response import get_gemini_response

def load_data(input_directory: Path, input_file_name: str) -> pd.DataFrame:
    df = pd.read_json(input_directory / input_file_name, lines=True)
    return df

def join_sentences(sentence_list: List[str]) -> str:
    """
    Join a list of sentences with alternating '\nA:' and '\nB:' prefixes.
    """
    joined_sentences = ''
    for i, sentence in enumerate(sentence_list):
        prefix = "\nA:" if i % 2 == 0 else "\nB:"
        joined_sentences += f"{prefix} {sentence}"
    return joined_sentences

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df['joined_bemba_sentences'] = df['bemba_sentences'].apply(join_sentences)
    df['joined_english_sentences'] = df['english_sentences'].apply(join_sentences)

    # Prompt to use for every line
    prompt = """
You are a helpful assistant that translates Bemba to English.

You will be given a short conversation between 2 Bemba speakers.
FYI, the first line in the conversation will always be describing some image that the speakers are looking at.

Your task is to translate the Bemba conversation into English.
Your translation should be as exact as possible, including all details.
It should be formatted as a conversation between two people, with each line starting with \nA: or \nB:. Just like the way you got it.
You should respond with the translation only, without any other text.

Here is the conversation for you to translate:
""".strip()
    df['full_translation_prompt'] = df['joined_bemba_sentences'].apply(
        lambda x: f"{prompt}\n{x}\nYour translation:"
    )
    return df

async def get_translations(
        df: pd.DataFrame, 
        llm_service, 
        output_directory: Path, 
        t_number: int=None,
        temperature: float=None
    ) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    if t_number is None:
        output_file_name = output_directory / f'big_c_conversations_test_{llm_service.__name__}.jsonl'
    else:
        output_file_name = output_directory / f'big_c_conversations_test_{llm_service.__name__}_t{t_number}.jsonl'

    for index, row in df.iterrows():
        print(f"Getting translation for index {index}")
        try:
            # Use 'joined_bemba_sentences' directly if using get_google_translate_response
            if llm_service == get_google_translate_response:
                input_text = row['joined_bemba_sentences']
            else:
                input_text = row['full_translation_prompt']
            
            if temperature is None:
                response = await llm_service(input_text)
            else:
                response = await llm_service(input_text, temperature=temperature)
            
            if t_number is None:
                df.at[index, f'{llm_service.__name__}_translation'] = response
            else:
                df.at[index, f'{llm_service.__name__}_translation_t{t_number}'] = response

            # Save the response incrementally to avoid losing progress
            with output_file_name.open('a') as f:
                json_line = json.dumps(df.loc[index].to_dict(), ensure_ascii=False)
                f.write(f'{json_line}\n')

        except Exception as e:
            print(f"Error processing index {index}: {e}")

def get_high_temp_translations(t_number: int=None) -> None:
    input_directory = Path('./Data/Static')
    input_file_name = 'big_c_conversations_test.jsonl'

    services = [
        (get_gemini_response, 3),
        (get_gemini_response, 4),
        (get_gemini_response, 5),
    ]
    for service, t_number in services:
        df = load_data(input_directory, input_file_name)
        df = prepare_dataframe(df)
        output_directory = Path('./Data/Output/translations_high_temp')
        asyncio.run(get_translations(
            df, 
            service, 
            output_directory,
            t_number=t_number
        ))

if __name__ == "__main__":
    get_high_temp_translations()
