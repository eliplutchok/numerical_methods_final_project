import pandas as pd
import json
import asyncio
import os

async def prepare_consistency_score_file(
        v1_model,
        v2_model,
        version_name="v0"
    ):

    v1_t1_jsonl_path = f"./Data/Output/translations_high_temp/big_c_conversations_test_{v1_model}_t1.jsonl"
    v1_t2_jsonl_path = f"./Data/Output/translations_high_temp/big_c_conversations_test_{v1_model}_t2.jsonl"
    v2_t1_jsonl_path = f"./Data/Output/translations_high_temp/big_c_conversations_test_{v2_model}_t1.jsonl"
    v2_t2_jsonl_path = f"./Data/Output/translations_high_temp/big_c_conversations_test_{v2_model}_t2.jsonl"

    df_1_t1 = pd.read_json(v1_t1_jsonl_path, lines=True)
    df_1_t2 = pd.read_json(v1_t2_jsonl_path, lines=True)
    df_2_t1 = pd.read_json(v2_t1_jsonl_path, lines=True)
    df_2_t2 = pd.read_json(v2_t2_jsonl_path, lines=True)
    
    result_df = df_1_t1
    result_df = result_df.merge(df_1_t2, on='id', how='left', suffixes=('', '_dup'))
    result_df = result_df.merge(df_2_t1, on='id', how='left', suffixes=('', '_dup'))
    result_df = result_df.merge(df_2_t2, on='id', how='left', suffixes=('', '_dup'))

    duplicate_columns = [col for col in result_df.columns if col.endswith('_dup')]
    result_df.drop(columns=duplicate_columns, inplace=True)
    
    result_df['v1_model'] = v1_model
    result_df['v2_model'] = v2_model

    output_file = f"./Data/Output/consistency_scores/{version_name}_big_c_test_{v1_model}_vs_{v2_model}.jsonl"
    
    # Check if the file already exists
    if os.path.exists(output_file):
        print(f"File '{output_file}' already exists. Skipping file creation.")
    else:
        # Save to JSONL file
        result_df.to_json(output_file, orient='records', lines=True)
        print(f"File '{output_file}' has been created.")

all_models = [
    "gemini_1_5_pro", 
    "o1_preview", 
    "gpt_4o",
    "sonnet_3_point_5"
]

model_pairs = []
for i in range(len(all_models)):
    for j in range(i+1, len(all_models)):
        model_pairs.append((all_models[i], all_models[j]))

print(model_pairs)

async def main():
    for v1_model, v2_model in model_pairs:
        await prepare_consistency_score_file(
            v1_model=v1_model,
            v2_model=v2_model,
        )

if __name__ == "__main__":
    asyncio.run(main())
