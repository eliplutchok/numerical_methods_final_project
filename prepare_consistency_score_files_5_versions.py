import pandas as pd
import json
import asyncio
import os

async def prepare_consistency_score_file(
        v1_model,
        v2_model,
        version_name="v0"
    ):

    translation_versions = ["t1", "t2", "t3", "t4", "t5"]

    def merge_model_translations(model):
        model_dfs = []
        for t_version in translation_versions:
            jsonl_path = f"./Data/Output/translations_high_temp/big_c_conversations_test_{model}_{t_version}.jsonl"
            df = pd.read_json(jsonl_path, lines=True)
            model_dfs.append(df)
        # Merge the dataframes
        model_df = model_dfs[0]
        for i in range(1, len(model_dfs)):
            model_df = model_df.merge(model_dfs[i], on='id', how='left', suffixes=('', '_dup'))
            # Remove duplicate columns
            duplicate_columns = [col for col in model_df.columns if col.endswith('_dup')]
            model_df.drop(columns=duplicate_columns, inplace=True)
        return model_df

    # Get merged dataframes for both models
    df_v1 = merge_model_translations(v1_model)
    df_v2 = merge_model_translations(v2_model)

    # Merge the two models' dataframes
    result_df = df_v1.merge(df_v2, on='id', how='left', suffixes=('', '_dup'))
    # Remove duplicate columns
    duplicate_columns = [col for col in result_df.columns if col.endswith('_dup')]
    result_df.drop(columns=duplicate_columns, inplace=True)
    
    result_df['v1_model'] = v1_model
    result_df['v2_model'] = v2_model

    output_file = f"./Data/Output/consistency_scores_5_versions/{version_name}_big_c_test_{v1_model}_vs_{v2_model}.jsonl"
    
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