import json
import random

# Paths to the input and output files
compiled_battles_file_path = "./Data/Output/prepared_files/compiled_battles.json"
output_elo_file_path = "./Data/Output/prepared_files/elo_rankings.json"

# Parameters for ELO calculation
INITIAL_ELO = 1000
K_FACTOR = 10  # Starting with a standard K-factor

def calculate_expected(player_rating, opponent_rating):
    """Calculate the expected score between two players."""
    return 1 / (1 + 10 ** ((opponent_rating - player_rating) / 400))

def update_elo(player_rating, opponent_rating, score, K):
    expected_score = calculate_expected(player_rating, opponent_rating)
    new_rating = player_rating + K * (score - expected_score)
    return new_rating

def compute_elo_for_battle_type(battle_type_data):
    """
    Compute ELO ratings for a given battle type.
    battle_type_data: Dictionary containing battles for the specific battle type.
    Returns a dictionary with model names as keys and their ELO ratings as values.
    """
    elo_ratings = {} 

    # Initialize ELO ratings for all models
    models = set()
    for battles in battle_type_data.values():
        for battle in battles:
            models.add(battle['model_1'])
            models.add(battle['model_2'])
    for model in models:
        elo_ratings[model] = INITIAL_ELO

    # Prepare a list of all individual matches
    matches = []
    for battles in battle_type_data.values():
        for battle in battles:
            model_1 = battle['model_1']
            model_2 = battle['model_2']
            winner = battle['winner']

            if winner == model_1:
                score = 1
            elif winner == model_2:
                score = 0
            else:  # 'tie' or any other value
                score = 0.5

            matches.append({
                'model_1': model_1,
                'model_2': model_2,
                'score': score
            })

    # Shuffle matches to randomize order
    random.shuffle(matches)

    # Update ELO ratings based on individual matches
    epochs = 20  # Number of times to iterate over all matches
    for _ in range(epochs):
        for match in matches:
            m1 = match['model_1']
            m2 = match['model_2']
            score = match['score']

            elo1 = elo_ratings[m1]
            elo2 = elo_ratings[m2]

            # Update ratings (you might use a dynamic K-factor here)
            new_elo1 = update_elo(elo1, elo2, score, K=K_FACTOR)
            new_elo2 = update_elo(elo2, elo1, 1 - score, K=K_FACTOR)

            elo_ratings[m1] = new_elo1
            elo_ratings[m2] = new_elo2

    return elo_ratings

def main():
    with open(compiled_battles_file_path, 'r') as f:
        compiled_battles = json.load(f)

    # Initialize a dictionary to store ELO rankings for each battle type
    elo_rankings = {}

    # List of battle types to process
    battle_types = [
        'judgment_battles',
        'bert_score_battles',
        'text_embedding_ada_002_battles',
        'consistency_battles',
        'consistency_scores_battles',
        'nv_embed_v2_battles',
        'consistency_scores_5_versions_battles'
    ]

    for battle_type in battle_types:
        print(f"Computing ELO rankings for {battle_type}...")

        # Extract battles for the current battle type
        battle_type_data = {}
        for id_str, battles_dict in compiled_battles.items():
            if battle_type in battles_dict:
                battle_type_data.setdefault(id_str, battles_dict[battle_type])

        # Compute ELO ratings for the battle type
        elo_ratings = compute_elo_for_battle_type(battle_type_data)

        # Sort the models based on ELO ratings
        sorted_elo = dict(sorted(elo_ratings.items(), key=lambda item: item[1], reverse=True))
        elo_rankings[battle_type] = sorted_elo

    # Write the ELO rankings to a JSON file
    with open(output_elo_file_path, 'w') as f:
        json.dump(elo_rankings, f, indent=2)

    print(f"ELO rankings have been computed and saved to {output_elo_file_path}")

if __name__ == "__main__":
    main()
