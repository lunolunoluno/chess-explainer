import os

import chess.pgn

from controller import Controller
from modules.utils import Debug, LLM

if __name__ == "__main__":
    dbg = Debug(debug=True)
    dbg.print("Debug: On")

    ctrl = Controller()
    # ctrl.save_good_comments_from_games()
    # ctrl.reformulate_good_comments()

    dataset_path = os.path.join(".", "data", "filtered_merged_20250708_134716.csv")
    ctrl.train_model(dataset_path, ["context", "moves"], "reformulated")

