# Backend/display_elo_rankings.py

import tkinter as tk
from tkinter import ttk
import pandas as pd
import json

# Path to the ELO rankings JSON file
elo_rankings_file_path = "./Data/Output/prepared_files/elo_rankings.json"

def load_elo_rankings():
    # Load the ELO rankings from the JSON file
    with open(elo_rankings_file_path, 'r') as f:
        elo_rankings = json.load(f)
    return elo_rankings

def prepare_data(elo_rankings):
    # Convert the ELO rankings into a DataFrame
    df_list = []
    for battle_type, rankings in elo_rankings.items():
        temp_df = pd.DataFrame(rankings.items(), columns=['Model', battle_type])
        df_list.append(temp_df)
    # Merge all DataFrames on the 'Model' column
    df = df_list[0]
    for temp_df in df_list[1:]:
        df = pd.merge(df, temp_df, on='Model', how='outer')
    # Reset index to include 'Model' as a column
    df.reset_index(drop=True, inplace=True)
    return df

class SortableTable(ttk.Treeview):
    def __init__(self, parent, df):
        # Define columns
        columns = df.columns.tolist()
        self.columns = columns
        super().__init__(parent, columns=columns, show='headings')
        self.df = df
        self._cached_items = None
        self.load_table()
        self.bind("<ButtonRelease-1>", self.on_click)

    def load_table(self):
        # Define headings and columns
        for col in self.columns:
            self.heading(col, text=col, command=lambda c=col: self.sort_by(c, False))
            self.column(col, anchor='center')

        # Insert data into the table
        for idx, row in self.df.iterrows():
            values = list(row)
            self.insert('', 'end', iid=idx, values=values)

    def sort_by(self, col, descending):
        # Get all data from the table
        data = [(self.set(child, col), child) for child in self.get_children('')]

        # Convert data to appropriate types
        try:
            # Try to convert to float
            data = [(float(value), child) for value, child in data]
        except ValueError:
            # Keep as string if conversion fails
            data = [(value, child) for value, child in data]

        # Sort the data
        data.sort(reverse=descending)

        # Rearrange items in sorted positions
        for index, (value, child) in enumerate(data):
            self.move(child, '', index)

        # Reverse sort next time
        self.heading(col, command=lambda: self.sort_by(col, not descending))

    def on_click(self, event):
        # Identify region and column
        region = self.identify('region', event.x, event.y)
        if region == 'heading':
            column_id = self.identify_column(event.x)
            col = self.column(column_id)['id']
            self.sort_by(col, False)

def main():
    elo_rankings = load_elo_rankings()
    df = prepare_data(elo_rankings)
    
    # Print the DataFrame to console
    print("\nELO Rankings:")
    print(df.to_string(index=False))
    
    # Create the GUI
    root = tk.Tk()
    root.title("ELO Ratings")
    # Create the sortable table
    table = SortableTable(root, df)
    table.pack(expand=True, fill='both')
    root.mainloop()

if __name__ == "__main__":
    main()