import os
import math
import chess.pgn

from chess.engine import PovScore, Cp, Mate
from enum import Enum

class WinningChanceJudgements(Enum):
    INACCURACY = 0.1
    MISTAKE = 0.2
    BLUNDER = 0.3

# Sources used to create this :
# https://github.com/lichess-org/lila/blob/cf9e10df24b767b3bc5ee3d88c45437ac722025d/modules/analyse/src/main/Advice.scala
class GameAnalyzer:
    def __init__(self, engine_path: str):
        assert os.path.exists(engine_path), f"{engine_path} doesn't exists !"

        self.engine_path = engine_path

    def analyze_game(self, game: chess.pgn.Game):
        with chess.engine.SimpleEngine.popen_uci(self.engine_path) as engine:
            board = game.board()
            prev_score = PovScore(Cp(0), turn=chess.WHITE)
            for move in game.mainline_moves():
                san_move = board.san(move)
                board.push(move)

                info = engine.analyse(board, chess.engine.Limit(depth=25, time=1.5))
                crt_score = info['score']
                print(f"{math.ceil(board.ply()/2)} {san_move} (cp={crt_score.white()}, depth={info['depth']})", end='')

                if isinstance(crt_score.relative, Cp) and isinstance(prev_score.relative, Cp):
                    crt_score_cp = crt_score.pov(board.turn).score()
                    prev_score_cp = prev_score.pov(board.turn).score()
                    delta = Cp(crt_score_cp).wdl().expectation() - Cp(prev_score_cp).wdl().expectation()

                    if delta >= WinningChanceJudgements.BLUNDER.value:
                        print("BLUNDER")
                    elif delta >= WinningChanceJudgements.MISTAKE.value:
                        print("MISTAKE")
                    elif delta >= WinningChanceJudgements.INACCURACY.value:
                        print("INACCURACY")
                    else:
                        print("")
                else:
                    print("")

                prev_score = crt_score


