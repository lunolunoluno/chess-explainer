import os
import chess.pgn

from modules.datautils import get_all_comments_after_error
from modules.gameanalyzer import GameAnalyzer
from modules.utils import Debug

if __name__ == "__main__":
    dbg = Debug(debug=True)
    dbg.print("This will print")

    dbg2 = Debug()
    dbg2.print("This will also print — same instance")
    print(dbg is dbg2)  # True — confirms singleton

    dbg.set_debug(False)
    dbg.print("This will NOT print")
    dbg.set_debug(True)

    filepath = os.path.join(".", "data", "quickgame.pgn")
    pgn = open(filepath)
    game = chess.pgn.read_game(pgn)

    ga = GameAnalyzer()
    ga.analyze_game(game=game)
    dbg.print(game)

    str_exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=True)
    comments = get_all_comments_after_error(game.accept(str_exporter))
    pgn.close()
    dbg.print(f"{len(comments)} comments found !")