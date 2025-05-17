import os
import math
import chess
import chess.pgn
import chess.engine
from enum import Enum

ENGINE_PATH = "/home/lucien/Documents/chess/stockfish17/src/stockfish"

# source : https://support.chess.com/en/articles/8572705-how-are-moves-classified-what-is-a-blunder-or-brilliant-etc
# the value associated with the classification is the minimum centipawn (cp) change for the move to receive such classification
class MoveClassification(Enum):
    INACCURACY = 50
    MISTAKE = 100
    BLUNDER = 200

def analyze_pgn_file(filepath: str):
    assert os.path.exists(filepath), f"{filepath} doesn't exists !"

    def get_classification(prev_move_eval:int, crt_move_eval:int)->MoveClassification|None:
        diff = abs(crt_move_eval - prev_move_eval)
        if diff >= MoveClassification.BLUNDER.value:
            return MoveClassification.BLUNDER
        elif diff >= MoveClassification.MISTAKE.value:
            return MoveClassification.MISTAKE
        elif diff >= MoveClassification.INACCURACY.value:
            return MoveClassification.INACCURACY
        return None

    pgn = open(filepath)
    game = chess.pgn.read_game(pgn)

    with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as engine:
        board = game.board()
        prev_eval = 0
        for move in game.mainline_moves():
            san_move = board.san(move)
            board.push(move)
            info = engine.analyse(board, chess.engine.Limit(depth=25, time=1.0))

            crt_eval = prev_eval
            match type(info['score'].relative):
                case chess.engine.Cp:
                    crt_eval = info['score'].white().score()
                case chess.engine.Mate:
                    crt_eval = 10000 if info['score'].turn == chess.WHITE else -10000

            classification = get_classification(prev_eval, crt_eval)
            print(f"Move: {san_move} ({crt_eval})", end=' ')
            match classification:
                case MoveClassification.BLUNDER:
                    print(f"is a BLUNDER")
                case MoveClassification.MISTAKE:
                    print(f"is a MISTAKE")
                case MoveClassification.INACCURACY:
                    print(f"is an INACCURACY")
                case _:
                    print("")
            prev_eval = crt_eval

if __name__ == "__main__":
    filepath = os.path.join(".", "data", "lildot_anatolym68mav_lichess.pgn")
    analyze_pgn_file(filepath)
