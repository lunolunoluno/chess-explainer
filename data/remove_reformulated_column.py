import os
import pandas as pd
from datetime import datetime


data_commented_path = os.path.join(".", "comments")

games = [
            os.path.join(data_commented_path, file)
            for file in os.listdir(data_commented_path)
            if os.path.isfile(os.path.join(data_commented_path, file)) and file.lower().endswith(".csv")
        ]

for game_index, game_comments in enumerate(games):
    try:
        df = pd.read_csv(game_comments)

        if 'reformulated' in df.columns:
            df = df.drop(columns=['reformulated'])
            df.to_csv(game_comments, index=False)
            print(f"Removed 'reformulated' column from: {game_comments}")
        else:
            print(f"No 'reformulated' column in: {game_comments}")

    except pd.errors.EmptyDataError:
        print(f"Skipped empty file: {game_comments}")