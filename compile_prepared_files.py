import os
import json
import pandas as pd

translations_folder = "./Data/Output/translations"
judgments_folder = "./Data/Output/judgments"
consistency_judgments_folder = "./Data/Output/consistency_judgments"
consistency_scores_folder = "./Data/Output/consistency_scores"
consistency_scores_5_versions_folder = "./Data/Output/consistency_scores_5_versions"
output_battles_file_path = "./Data/Output/prepared_files/compiled_battles.json"
output_totals_file_path = "./Data/Output/prepared_files/battle_totals.json"

# Load all translations and store bert scores and similarities per model per id
translation_file_prefix = "a_v2_big_c_conversations_test_"
translation_files = os.listdir(translations_folder)

output_folder = "./Data/Output/prepared_files"

# ------------------------------------------------------------------------------------------------

overall_data_dict = {}
for filename in translation_files:
    if filename.endswith(".jsonl") and filename.startswith(translation_file_prefix):
        model_name = filename.replace(f"{translation_file_prefix}", "").replace(".jsonl", "")

        fields_to_keep = [
            "id", 
            "joined_bemba_sentences",
            "joined_english_sentences",
            f"{model_name}_translation",
            f"{model_name}_bertscore",
            f"{model_name}_text-embedding-ada-002_similarity",
            f"{model_name}_nv-embed-v2_similarity"
        ]
        
        data = pd.read_json(os.path.join(translations_folder, filename), lines=True)
        # Convert DataFrame to a list of dictionaries
        data_records = data.to_dict(orient='records')
        data_records = [record for record in data_records if all(field in record for field in fields_to_keep)]
        # convert to data dict with the ids as keys
        data_dict = {record["id"]: record for record in data_records}
        model_package = {
            "model_name": model_name,
            "data": data_dict
        }
        overall_data_dict[model_name] = model_package
        output_file_path = os.path.join(output_folder, f"{model_name}.json")
        with open(output_file_path, "w") as f:
            json.dump(model_package, f)

# ------------------------------------------------------------------------------------------------

# Data structure to store bert scores and similarities
model_scores = {}  # model_name -> {id_str -> {'bert_score': ..., 'similarity': ...}}

for filename in translation_files:
    if filename.endswith(".jsonl") and filename.startswith(translation_file_prefix):
        model_name = filename.replace(f"{translation_file_prefix}", "").replace(".jsonl", "")
        model_scores[model_name] = {}
        with open(os.path.join(translations_folder, filename), 'r') as f:
            for line in f:
                data = json.loads(line)
                id_str = str(data['id'])
                bert_score = data.get(f"{model_name}_bertscore")
                similarity = data.get(f"{model_name}_text-embedding-ada-002_similarity")
                nv_similarity = data.get(f"{model_name}_nv-embed-v2_similarity")
                model_scores[model_name][id_str] = {
                    'bert_score': bert_score,
                    'similarity': similarity,
                    'nv_similarity': nv_similarity
                }

# Data structure to store compiled battles
compiled_battles = {}

# Initialize totals data structure with detailed opponent info
model_names = list(model_scores.keys())
battle_types = [
    'judgment_battles', 
    'bert_score_battles', 
    'text_embedding_ada_002_battles', 
    'nv_embed_v2_battles', 
    'consistency_battles', 
    'consistency_scores_battles',
    'consistency_scores_5_versions_battles'
]
battle_totals = {
    battle_type: {
        model: {
            'total_wins': 0,
            'total_losses': 0,
            'total_ties': 0,
            'wins_against': {},
            'losses_against': {},
            'ties_against': {}
        } for model in model_names
    } for battle_type in battle_types
}

# Process judgment files
judgment_files = os.listdir(judgments_folder)
judgment_file_prefix = "v2_big_c_test_"

for filename in judgment_files:
    if filename.endswith(".jsonl") and filename.startswith(judgment_file_prefix):
        # Extract model names from filename
        model_pair = filename.replace(judgment_file_prefix, "").replace(".jsonl", "")
        model1_name, model2_name = model_pair.split("_vs_", 1)
        with open(os.path.join(judgments_folder, filename), 'r') as f:
            for line in f:
                data = json.loads(line)
                id_str = str(data['id'])
                # Initialize the id in compiled_battles if not already present
                if id_str not in compiled_battles:
                    compiled_battles[id_str] = {
                        'judgment_battles': [],
                        'bert_score_battles': [],
                        'text_embedding_ada_002_battles': [],
                        'nv_embed_v2_battles': [],
                        'consistency_battles': [],
                        'consistency_scores_battles': [],
                        'consistency_scores_5_versions_battles': []
                    }
                # Get the winner from 'gpt_4o_judgment'
                winner_model = data.get('gpt_4o_judgment')
                # Append to judgment_battles
                compiled_battles[id_str]['judgment_battles'].append({
                    'model_1': model1_name,
                    'model_2': model2_name,
                    'winner': winner_model
                })
                # Initialize opponent stats if not present
                for model, opponent in [(model1_name, model2_name), (model2_name, model1_name)]:
                    for battle_type in battle_types:
                        if opponent not in battle_totals[battle_type][model]['wins_against']:
                            battle_totals[battle_type][model]['wins_against'][opponent] = 0
                            battle_totals[battle_type][model]['losses_against'][opponent] = 0
                            battle_totals[battle_type][model]['ties_against'][opponent] = 0
                # Update totals for judgment battles
                if winner_model == model1_name:
                    battle_totals['judgment_battles'][model1_name]['total_wins'] += 1
                    battle_totals['judgment_battles'][model1_name]['wins_against'][model2_name] += 1
                    battle_totals['judgment_battles'][model2_name]['total_losses'] += 1
                    battle_totals['judgment_battles'][model2_name]['losses_against'][model1_name] += 1
                elif winner_model == model2_name:
                    battle_totals['judgment_battles'][model2_name]['total_wins'] += 1
                    battle_totals['judgment_battles'][model2_name]['wins_against'][model1_name] += 1
                    battle_totals['judgment_battles'][model1_name]['total_losses'] += 1
                    battle_totals['judgment_battles'][model1_name]['losses_against'][model2_name] += 1
                else:  # tie or invalid winner
                    battle_totals['judgment_battles'][model1_name]['total_ties'] += 1
                    battle_totals['judgment_battles'][model1_name]['ties_against'][model2_name] += 1
                    battle_totals['judgment_battles'][model2_name]['total_ties'] += 1
                    battle_totals['judgment_battles'][model2_name]['ties_against'][model1_name] += 1

                # Get bert scores for both models
                bert_score_1 = model_scores.get(model1_name, {}).get(id_str, {}).get('bert_score')
                bert_score_2 = model_scores.get(model2_name, {}).get(id_str, {}).get('bert_score')
                if bert_score_1 is not None and bert_score_2 is not None:
                    if bert_score_1 > bert_score_2:
                        winner_bert = model1_name
                    elif bert_score_1 < bert_score_2:
                        winner_bert = model2_name
                    else:
                        winner_bert = 'tie'
                    compiled_battles[id_str]['bert_score_battles'].append({
                        'model_1': model1_name,
                        'model_2': model2_name,
                        'winner': winner_bert
                    })
                    # Update totals for bert score battles
                    if winner_bert == model1_name:
                        battle_totals['bert_score_battles'][model1_name]['total_wins'] += 1
                        battle_totals['bert_score_battles'][model1_name]['wins_against'][model2_name] += 1
                        battle_totals['bert_score_battles'][model2_name]['total_losses'] += 1
                        battle_totals['bert_score_battles'][model2_name]['losses_against'][model1_name] += 1
                    elif winner_bert == model2_name:
                        battle_totals['bert_score_battles'][model2_name]['total_wins'] += 1
                        battle_totals['bert_score_battles'][model2_name]['wins_against'][model1_name] += 1
                        battle_totals['bert_score_battles'][model1_name]['total_losses'] += 1
                        battle_totals['bert_score_battles'][model1_name]['losses_against'][model2_name] += 1
                    else:  # tie
                        battle_totals['bert_score_battles'][model1_name]['total_ties'] += 1
                        battle_totals['bert_score_battles'][model1_name]['ties_against'][model2_name] += 1
                        battle_totals['bert_score_battles'][model2_name]['total_ties'] += 1
                        battle_totals['bert_score_battles'][model2_name]['ties_against'][model1_name] += 1

                # Get text embedding similarities for both models
                sim_1 = model_scores.get(model1_name, {}).get(id_str, {}).get('similarity')
                sim_2 = model_scores.get(model2_name, {}).get(id_str, {}).get('similarity')
                if sim_1 is not None and sim_2 is not None:
                    if sim_1 > sim_2:
                        winner_sim = model1_name
                    elif sim_1 < sim_2:
                        winner_sim = model2_name
                    else:
                        winner_sim = 'tie'
                    compiled_battles[id_str]['text_embedding_ada_002_battles'].append({
                        'model_1': model1_name,
                        'model_2': model2_name,
                        'winner': winner_sim
                    })
                    # Update totals for similarity battles
                    if winner_sim == model1_name:
                        battle_totals['text_embedding_ada_002_battles'][model1_name]['total_wins'] += 1
                        battle_totals['text_embedding_ada_002_battles'][model1_name]['wins_against'][model2_name] += 1
                        battle_totals['text_embedding_ada_002_battles'][model2_name]['total_losses'] += 1
                        battle_totals['text_embedding_ada_002_battles'][model2_name]['losses_against'][model1_name] += 1
                    elif winner_sim == model2_name:
                        battle_totals['text_embedding_ada_002_battles'][model2_name]['total_wins'] += 1
                        battle_totals['text_embedding_ada_002_battles'][model2_name]['wins_against'][model1_name] += 1
                        battle_totals['text_embedding_ada_002_battles'][model1_name]['total_losses'] += 1
                        battle_totals['text_embedding_ada_002_battles'][model1_name]['losses_against'][model2_name] += 1
                    else:  # tie
                        battle_totals['text_embedding_ada_002_battles'][model1_name]['total_ties'] += 1
                        battle_totals['text_embedding_ada_002_battles'][model1_name]['ties_against'][model2_name] += 1
                        battle_totals['text_embedding_ada_002_battles'][model2_name]['total_ties'] += 1
                        battle_totals['text_embedding_ada_002_battles'][model2_name]['ties_against'][model1_name] += 1

                # Get nv embeddings similarities for both models
                nv_sim_1 = model_scores.get(model1_name, {}).get(id_str, {}).get('nv_similarity')
                nv_sim_2 = model_scores.get(model2_name, {}).get(id_str, {}).get('nv_similarity')
                if nv_sim_1 is not None and nv_sim_2 is not None:
                    if nv_sim_1 > nv_sim_2:
                        winner_nv = model1_name
                    elif nv_sim_1 < nv_sim_2:
                        winner_nv = model2_name
                    else:
                        winner_nv = 'tie'
                    compiled_battles[id_str]['nv_embed_v2_battles'].append({
                        'model_1': model1_name,
                        'model_2': model2_name,
                        'winner': winner_nv
                    })
                    # Update totals for nv embeddings battles
                    if winner_nv == model1_name:
                        battle_totals['nv_embed_v2_battles'][model1_name]['total_wins'] += 1
                        battle_totals['nv_embed_v2_battles'][model1_name]['wins_against'][model2_name] += 1
                        battle_totals['nv_embed_v2_battles'][model2_name]['total_losses'] += 1
                        battle_totals['nv_embed_v2_battles'][model2_name]['losses_against'][model1_name] += 1
                    elif winner_nv == model2_name:
                        battle_totals['nv_embed_v2_battles'][model2_name]['total_wins'] += 1
                        battle_totals['nv_embed_v2_battles'][model2_name]['wins_against'][model1_name] += 1
                        battle_totals['nv_embed_v2_battles'][model1_name]['total_losses'] += 1
                        battle_totals['nv_embed_v2_battles'][model1_name]['losses_against'][model2_name] += 1
                    else:  # tie
                        battle_totals['nv_embed_v2_battles'][model1_name]['total_ties'] += 1
                        battle_totals['nv_embed_v2_battles'][model1_name]['ties_against'][model2_name] += 1
                        battle_totals['nv_embed_v2_battles'][model2_name]['total_ties'] += 1
                        battle_totals['nv_embed_v2_battles'][model2_name]['ties_against'][model1_name] += 1

# Process consistency judgments
consistency_files = os.listdir(consistency_judgments_folder)
consistency_file_prefix = "v1_big_c_test_"

for filename in consistency_files:
    if filename.endswith(".jsonl") and filename.startswith(consistency_file_prefix):
        # Extract model names from filename
        model_pair = filename.replace(consistency_file_prefix, "").replace(".jsonl", "")
        model1_name, model2_name = model_pair.split("_vs_", 1)
        with open(os.path.join(consistency_judgments_folder, filename), 'r') as f:
            for line in f:
                data = json.loads(line)
                id_str = str(data['id'])
                # Initialize the id in compiled_battles if not already present
                if id_str not in compiled_battles:
                    compiled_battles[id_str] = {
                        'judgment_battles': [],
                        'bert_score_battles': [],
                        'text_embedding_ada_002_battles': [],
                        'nv_embed_v2_battles': [],
                        'consistency_battles': [],
                        'consistency_scores_battles': [],
                        'consistency_scores_5_versions_battles': []
                    }
                # Get the winner from 'gpt_4o_consistency_judgment'
                winner_consistency = data.get('gpt_4o_consistency_judgment')
                # Append to consistency_battles
                compiled_battles[id_str]['consistency_battles'].append({
                    'model_1': model1_name,
                    'model_2': model2_name,
                    'winner': winner_consistency
                })
                # Initialize opponent stats if not present
                for model, opponent in [(model1_name, model2_name), (model2_name, model1_name)]:
                    for battle_type in ['consistency_battles']:
                        if opponent not in battle_totals[battle_type][model]['wins_against']:
                            battle_totals[battle_type][model]['wins_against'][opponent] = 0
                            battle_totals[battle_type][model]['losses_against'][opponent] = 0
                            battle_totals[battle_type][model]['ties_against'][opponent] = 0
                # Update totals for consistency battles
                if winner_consistency == model1_name:
                    battle_totals['consistency_battles'][model1_name]['total_wins'] += 1
                    battle_totals['consistency_battles'][model1_name]['wins_against'][model2_name] += 1
                    battle_totals['consistency_battles'][model2_name]['total_losses'] += 1
                    battle_totals['consistency_battles'][model2_name]['losses_against'][model1_name] += 1
                elif winner_consistency == model2_name:
                    battle_totals['consistency_battles'][model2_name]['total_wins'] += 1
                    battle_totals['consistency_battles'][model2_name]['wins_against'][model1_name] += 1
                    battle_totals['consistency_battles'][model1_name]['total_losses'] += 1
                    battle_totals['consistency_battles'][model1_name]['losses_against'][model2_name] += 1
                else:  # tie
                    battle_totals['consistency_battles'][model1_name]['total_ties'] += 1
                    battle_totals['consistency_battles'][model1_name]['ties_against'][model2_name] += 1
                    battle_totals['consistency_battles'][model2_name]['total_ties'] += 1
                    battle_totals['consistency_battles'][model2_name]['ties_against'][model1_name] += 1

# Process consistency scores
consistency_scores_files = os.listdir(consistency_scores_folder)

for filename in consistency_scores_files:
    if filename.endswith(".jsonl"):
        with open(os.path.join(consistency_scores_folder, filename), 'r') as f:
            for line in f:
                data = json.loads(line)
                id_str = str(data['id'])
                # Initialize the id in compiled_battles if not already present
                if id_str not in compiled_battles:
                    compiled_battles[id_str] = {
                        'judgment_battles': [],
                        'bert_score_battles': [],
                        'text_embedding_ada_002_battles': [],
                        'nv_embed_v2_battles': [],
                        'consistency_battles': [],
                        'consistency_scores_battles': [],
                        'consistency_scores_5_versions_battles': []
                    }
                # Get the winner from 'text-embedding-ada-002_consistency_judgment'
                winner_model = data.get('text-embedding-ada-002_consistency_judgment')
                if not winner_model:
                    winner_model = None
                # Extract models from columns ending with '_t1_t2_text-embedding-ada-002_similarity'
                model_columns = [col for col in data.keys() if col.endswith('_t1_t2_text-embedding-ada-002_similarity')]
                models_involved = [col.replace('_t1_t2_text-embedding-ada-002_similarity', '') for col in model_columns]
                if len(models_involved) == 2:
                    model1_name, model2_name = models_involved
                else:
                    # Handle cases where the number of models is not 2
                    continue
                # Append to consistency_scores_battles
                compiled_battles[id_str]['consistency_scores_battles'].append({
                    'model_1': model1_name,
                    'model_2': model2_name,
                    'winner': winner_model
                })
                # Ensure models are in battle_totals
                for model in [model1_name, model2_name]:
                    if model not in battle_totals['consistency_scores_battles']:
                        battle_totals['consistency_scores_battles'][model] = {
                            'total_wins': 0,
                            'total_losses': 0,
                            'total_ties': 0,
                            'wins_against': {},
                            'losses_against': {},
                            'ties_against': {}
                        }
                # Initialize opponent stats if not present
                for model, opponent in [(model1_name, model2_name), (model2_name, model1_name)]:
                    stats = battle_totals['consistency_scores_battles'][model]
                    if opponent not in stats['wins_against']:
                        stats['wins_against'][opponent] = 0
                        stats['losses_against'][opponent] = 0
                        stats['ties_against'][opponent] = 0
                # Update totals for consistency scores battles
                if winner_model == model1_name:
                    battle_totals['consistency_scores_battles'][model1_name]['total_wins'] += 1
                    battle_totals['consistency_scores_battles'][model1_name]['wins_against'][model2_name] += 1
                    battle_totals['consistency_scores_battles'][model2_name]['total_losses'] += 1
                    battle_totals['consistency_scores_battles'][model2_name]['losses_against'][model1_name] += 1
                elif winner_model == model2_name:
                    battle_totals['consistency_scores_battles'][model2_name]['total_wins'] += 1
                    battle_totals['consistency_scores_battles'][model2_name]['wins_against'][model1_name] += 1
                    battle_totals['consistency_scores_battles'][model1_name]['total_losses'] += 1
                    battle_totals['consistency_scores_battles'][model1_name]['losses_against'][model2_name] += 1
                else:  # tie or invalid winner
                    battle_totals['consistency_scores_battles'][model1_name]['total_ties'] += 1
                    battle_totals['consistency_scores_battles'][model1_name]['ties_against'][model2_name] += 1
                    battle_totals['consistency_scores_battles'][model2_name]['total_ties'] += 1
                    battle_totals['consistency_scores_battles'][model2_name]['ties_against'][model1_name] += 1

# Process consistency scores 5 versions
consistency_scores_5_versions_files = os.listdir(consistency_scores_5_versions_folder)

for filename in consistency_scores_5_versions_files:
    if filename.endswith(".jsonl"):
        with open(os.path.join(consistency_scores_5_versions_folder, filename), 'r') as f:
            for line in f:
                data = json.loads(line)
                id_str = str(data['id'])
                # Initialize the id in compiled_battles if not already present
                if id_str not in compiled_battles:
                    compiled_battles[id_str] = {
                        'judgment_battles': [],
                        'bert_score_battles': [],
                        'text_embedding_ada_002_battles': [],
                        'nv_embed_v2_battles': [],
                        'consistency_battles': [],
                        'consistency_scores_battles': [],
                        'consistency_scores_5_versions_battles': []
                    }
                # Get the winner from 'text-embedding-ada-002_consistency_judgment_5_versions'
                winner_model = data.get('text-embedding-ada-002_consistency_judgment_5_versions')
                if not winner_model:
                    winner_model = None
                
                model_columns = [col for col in data.keys() if col.endswith('_text-embedding-ada-002_consistency_score')]
                models_involved = [col.replace('_text-embedding-ada-002_consistency_score', '') for col in model_columns]
                if len(models_involved) == 2:
                    model1_name, model2_name = models_involved
                else:
                    # Handle cases where the number of models is not 2
                    continue
                # Append to consistency_scores_5_versions_battles
                compiled_battles[id_str]['consistency_scores_5_versions_battles'].append({
                    'model_1': model1_name,
                    'model_2': model2_name,
                    'winner': winner_model
                })
                # Ensure models are in battle_totals
                for model in [model1_name, model2_name]:
                    if model not in battle_totals['consistency_scores_5_versions_battles']:
                        battle_totals['consistency_scores_5_versions_battles'][model] = {
                            'total_wins': 0,
                            'total_losses': 0,
                            'total_ties': 0,
                            'wins_against': {},
                            'losses_against': {},
                            'ties_against': {}
                        }
                # Initialize opponent stats if not present
                for model, opponent in [(model1_name, model2_name), (model2_name, model1_name)]:
                    stats = battle_totals['consistency_scores_5_versions_battles'][model]
                    if opponent not in stats['wins_against']:
                        stats['wins_against'][opponent] = 0
                        stats['losses_against'][opponent] = 0
                        stats['ties_against'][opponent] = 0
                # Update totals for consistency scores battles
                if winner_model == model1_name:
                    battle_totals['consistency_scores_5_versions_battles'][model1_name]['total_wins'] += 1
                    battle_totals['consistency_scores_5_versions_battles'][model1_name]['wins_against'][model2_name] += 1
                    battle_totals['consistency_scores_5_versions_battles'][model2_name]['total_losses'] += 1
                    battle_totals['consistency_scores_5_versions_battles'][model2_name]['losses_against'][model1_name] += 1
                elif winner_model == model2_name:
                    battle_totals['consistency_scores_5_versions_battles'][model2_name]['total_wins'] += 1
                    battle_totals['consistency_scores_5_versions_battles'][model2_name]['wins_against'][model1_name] += 1
                    battle_totals['consistency_scores_5_versions_battles'][model1_name]['total_losses'] += 1
                    battle_totals['consistency_scores_5_versions_battles'][model1_name]['losses_against'][model2_name] += 1
                else:  # tie or invalid winner
                    battle_totals['consistency_scores_5_versions_battles'][model1_name]['total_ties'] += 1
                    battle_totals['consistency_scores_5_versions_battles'][model1_name]['ties_against'][model2_name] += 1
                    battle_totals['consistency_scores_5_versions_battles'][model2_name]['total_ties'] += 1
                    battle_totals['consistency_scores_5_versions_battles'][model2_name]['ties_against'][model1_name] += 1

# Write the compiled battles to a JSON file
with open(output_battles_file_path, 'w') as f:
    json.dump(compiled_battles, f, indent=2)

# Write the battle totals to a JSON file
with open(output_totals_file_path, 'w') as f:
    json.dump(battle_totals, f, indent=2)