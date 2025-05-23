import os
import chess.pgn

from modules.datautils import get_all_comments_after_error

if __name__ == "__main__":
    # https://lichess.org/trax6sXR/white
    filepath = os.path.join(".", "data", "lildot_anatolym68mav_lichess_commented.pgn")

    str_exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=True)
    pgn = open(filepath)
    game = chess.pgn.read_game(pgn)
    comments = get_all_comments_after_error(game.accept(str_exporter))
    pgn.close()

    print(f"{len(comments)} comments found !")

