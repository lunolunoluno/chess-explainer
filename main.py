import os

import chess

from controller import Controller
from modules.gameanalyzer import GameAnalyzer
from modules.utils import Debug

if __name__ == "__main__":
    dbg = Debug(debug=True)
    dbg.print("Debug: On")

    ctrl = Controller()
    ctrl.analyze_annotated_games()

    # filepath = os.path.join(".", "data", "raw", "quickgame.pgn")
    # ga = GameAnalyzer()
    # pgn = open(filepath)
    # game = chess.pgn.read_game(pgn)
    # ga.analyze_game(game=game)
    # print(game)
