import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import tkinter as tk
from tkinter import ttk

# Path to the battle totals JSON file
battle_totals_file_path = "./Data/Output/prepared_files/battle_totals.json"

def load_battle_totals():
    # Load the battle totals from the JSON file
    with open(battle_totals_file_path, 'r') as f:
        battle_totals = json.load(f)
    return battle_totals

def prepare_heatmap_data(battle_totals, battle_type='judgment_battles'):
    # Extract the models
    models = sorted(battle_totals[battle_type].keys())
    # Initialize DataFrames
    heatmap_data = pd.DataFrame(0, index=models, columns=models)
    hover_text = pd.DataFrame('', index=models, columns=models)
    # Populate the DataFrames
    for model in models:
        wins = battle_totals[battle_type][model]['wins_against']
        losses = battle_totals[battle_type][model]['losses_against']
        ties = battle_totals[battle_type][model]['ties_against']
        for opponent in models:
            win_count = wins.get(opponent, 0)
            loss_count = losses.get(opponent, 0)
            tie_count = ties.get(opponent, 0)
            total_battles = win_count + loss_count + tie_count
            # Assign the number of wins to the heatmap
            heatmap_data.loc[model, opponent] = win_count
            # Create hover text
            hover_text.loc[model, opponent] = (
                f"Wins vs {opponent}: {win_count}<br>"
                f"Losses vs {opponent}: {loss_count}<br>"
                f"Ties vs {opponent}: {tie_count}<br>"
                f"Total Battles: {total_battles}"
            )
    return heatmap_data, hover_text

def plot_interactive_heatmap(heatmap_data, hover_text, battle_type='judgment_battles'):
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Opponent", y="Model", color="Wins"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale="Viridis",
        text_auto=True,
        aspect="auto"
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>%{customdata}",
        customdata=hover_text.values
    )
    fig.update_layout(
        title=f"Battle Totals Heatmap - {battle_type.replace('_', ' ').title()}",
        xaxis_nticks=len(heatmap_data.columns),
        yaxis_nticks=len(heatmap_data.index),
    )
    fig.show()

def on_button_click(battle_type, battle_totals):
    heatmap_data, hover_text = prepare_heatmap_data(battle_totals, battle_type)
    plot_interactive_heatmap(heatmap_data, hover_text, battle_type)

def plot_battle_network(battle_totals, battle_type='judgment_battles'):
    # Create directed graph
    G = nx.DiGraph()

    # Add nodes and edges
    models = battle_totals[battle_type].keys()
    for model in models:
        G.add_node(model)
    for model in models:
        wins = battle_totals[battle_type][model]['wins_against']
        for opponent, count in wins.items():
            if count > 0:
                G.add_edge(model, opponent, weight=count)

    # Draw the graph
    pos = nx.spring_layout(G, k=2)
    edge_weights = [G[u][v]['weight'] for u,v in G.edges()]
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', arrows=True, width=edge_weights)
    plt.title(f"Battle Network - {battle_type.replace('_', ' ').title()}")
    plt.show()

def plot_battle_sankey(battle_totals, battle_type='judgment_battles'):
    # Prepare data
    source = []
    target = []
    value = []

    models = battle_totals[battle_type].keys()
    for model in models:
        wins = battle_totals[battle_type][model]['wins_against']
        for opponent, count in wins.items():
            if count > 0:
                source.append(model)
                target.append(opponent)
                value.append(count)

    # Assign indices to models
    model_indices = {model: idx for idx, model in enumerate(models)}

    # Map model names to indices
    source_indices = [model_indices[s] for s in source]
    target_indices = [model_indices[t] for t in target]

    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = list(models),
        ),
        link = dict(
          source = source_indices,
          target = target_indices,
          value = value,
      ))])

    fig.update_layout(title_text=f"Battle Sankey Diagram - {battle_type.replace('_', ' ').title()}", font_size=10)
    fig.show()

def plot_pyvis_network(battle_totals, battle_type='judgment_battles'):
    # Create a PyVis network
    net = Network(notebook=False)

    # Add nodes
    models = battle_totals[battle_type].keys()
    for model in models:
        net.add_node(model, label=model)

    # Add edges
    for model in models:
        wins = battle_totals[battle_type][model]['wins_against']
        for opponent, count in wins.items():
            if count > 0:
                net.add_edge(model, opponent, value=count, title=f"Wins: {count}")

    # Customize options
    net.set_options("""
    var options = {
      "edges": {
        "arrows": {
          "to": {
            "enabled": true
          }
        },
        "color": {
          "inherit": true
        },
        "smooth": false
      },
      "nodes": {
        "color": {
          "highlight": {
            "border": "rgba(0,0,0,1)",
            "background": "rgba(255,0,0,1)"
          }
        },
        "font": {
          "size": 16
        }
      },
      "physics": {
        "enabled": true
      }
    }
    """)

    # Generate and show the network
    net.show(f"battle_network_{battle_type}.html")

def main():
    battle_totals = load_battle_totals()
    # Create the GUI
    root = tk.Tk()
    root.title("Interactive Battle Totals Heatmap")
    # Create buttons for each battle type
    battle_types = list(battle_totals.keys())
    for battle_type in battle_types:
        button_text = battle_type.replace('_', ' ').title()
        btn = ttk.Button(
            root,
            text=button_text,
            command=lambda bt=battle_type: on_button_click(bt, battle_totals)
        )
        btn.pack(pady=5, padx=10, fill='x')
    root.mainloop()

if __name__ == "__main__":
    main()
