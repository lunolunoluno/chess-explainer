import os
import chess.pgn

from gameanalyzer import GameAnalyzer

if __name__ == "__main__":
    # https://lichess.org/trax6sXR/white
    filepath = os.path.join(".", "data", "lildot_anatolym68mav_lichess.pgn")

    ga = GameAnalyzer()

    pgn = open(filepath)
    game = chess.pgn.read_game(pgn)

    ga.analyze_game(game=game)

