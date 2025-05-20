import os
import chess.pgn

from gameanalyzer import GameAnalyzer

# downloaded from https://stockfishchess.org/download/
ENGINE_PATH = "/home/lucien/Documents/chess/stockfish17/src/stockfish"

if __name__ == "__main__":
    # https://lichess.org/trax6sXR/white
    filepath = os.path.join(".", "data", "lildot_anatolym68mav_lichess.pgn")

    ga = GameAnalyzer(engine_path=ENGINE_PATH)

    pgn = open(filepath)
    game = chess.pgn.read_game(pgn)

    ga.analyze_game(game=game)

