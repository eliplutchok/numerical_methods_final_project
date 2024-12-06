import json
from sklearn.metrics import cohen_kappa_score
from itertools import combinations
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
with open('./Data/Output/prepared_files/compiled_battles.json', 'r') as f:
    data = json.load(f)

all_models = [
    'o1_preview', 
    'gpt_4o', 
    'claude_3_5_sonnet', 
    'gemini_1_5_pro'
]

# List of test names in your data
test_names = [
    "judgment_battles",
    "bert_score_battles",
    "text_embedding_ada_002_battles",
    "consistency_battles",
    "consistency_scores_battles",
    "nv_embed_v2_battles",
    "consistency_scores_5_versions_battles"
]

# Dictionary to store outcomes for each test
# Format: { test_name: { battle_id: outcome } }
# where battle_id is a tuple (model_1, model_2, example_id)
test_outcomes = {test_name: {} for test_name in test_names}

# Iterate over each example
for example_id, example_data in data.items():
    # Iterate over each test
    for test_name in test_names:
        battles = example_data.get(test_name, [])
        for battle in battles:
            models = sorted([battle['model_1'], battle['model_2']])
            model_1, model_2 = models[0], models[1]
            # Check if both models are in all_models list
            if model_1 in all_models and model_2 in all_models:
                battle_id = (model_1, model_2, example_id)
                # Determine the outcome
                if battle['winner'] == model_1:
                    outcome = 1
                elif battle['winner'] == model_2:
                    outcome = 0
                else:
                    continue  # Skip ties or unrecognized winners
                # Store the outcome
                test_outcomes[test_name][battle_id] = outcome

# Function to compute Cohen's Kappa between two tests
def compute_kappa(test1_outcomes, test2_outcomes):
    # Find common battle IDs
    common_ids = set(test1_outcomes.keys()) & set(test2_outcomes.keys())
    if not common_ids:
        return None  # No common battles to compare
    # Extract the outcomes
    outcomes1 = [test1_outcomes[battle_id] for battle_id in common_ids]
    outcomes2 = [test2_outcomes[battle_id] for battle_id in common_ids]
    # Compute Cohen's Kappa
    kappa = cohen_kappa_score(outcomes1, outcomes2)
    return kappa

# Compute Cohen's Kappa between each pair of tests and store in a matrix
kappa_matrix = pd.DataFrame(index=test_names, columns=test_names, dtype=float)

for test1 in test_names:
    for test2 in test_names:
        if test1 == test2:
            kappa_matrix.loc[test1, test2] = 1.0  # Perfect agreement with itself
        else:
            kappa = compute_kappa(test_outcomes[test1], test_outcomes[test2])
            if kappa is not None:
                kappa_matrix.loc[test1, test2] = kappa
            else:
                kappa_matrix.loc[test1, test2] = None  # No common battles

# Print the kappa matrix
print("\nCohen's Kappa Matrix:")
print(kappa_matrix)

# Create a mapping for display names (aliases)
display_names = {
    "judgment_battles": "llm_judgements",
    "bert_score_battles": "bert_score",
    "text_embedding_ada_002_battles": "ada_embedding",
    "consistency_battles": "consistency_llm_judgments",
    "consistency_scores_battles": "consistency_scores",
    "nv_embed_v2_battles": "nv_embed_v2",
    "consistency_scores_5_versions_battles": "avg_consistency_scores"
}

# Update the indices and columns of the matrix for plotting
kappa_matrix_display = kappa_matrix.copy().round(2)  # Round to 2 decimal places
kappa_matrix_display.index = [display_names[name] for name in kappa_matrix.index]
kappa_matrix_display.columns = [display_names[name] for name in kappa_matrix.columns]

# Set pandas display options to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Print the kappa matrix
print("\nCohen's Kappa Matrix:")
print(kappa_matrix_display)

# Plotting the heatmap with blue-white colormap
plt.figure(figsize=(10, 8))
sns.heatmap(
    kappa_matrix.astype(float),
    annot=True,
    fmt=".2f",
    cmap="Blues",
    vmin=0,
    vmax=1,
    xticklabels=[display_names[name] for name in test_names],
    yticklabels=[display_names[name] for name in test_names]
)
plt.title("Cohen's Kappa Heatmap between Tests")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
