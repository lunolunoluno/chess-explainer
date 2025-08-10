import os
import math
import chess.pgn

from enum import Enum
from dotenv import load_dotenv
from chess.engine import PovScore, Cp, Mate, MateGiven
from chess.pgn import NAG_BLUNDER, NAG_MISTAKE, NAG_DUBIOUS_MOVE

from modules.utils import Debug

load_dotenv()


class WinningChanceJudgements(Enum):
    INACCURACY = 0.1
    MISTAKE = 0.2
    BLUNDER = 0.3


# Sources used to create this :
# https://github.com/lichess-org/lila/blob/cf9e10df24b767b3bc5ee3d88c45437ac722025d/modules/analyse/src/main/Advice.scala
class GameAnalyzer:
    def __init__(self) -> None:
        self.dbg = Debug()

        engine_path = os.getenv("ENGINE_PATH")
        assert os.path.exists(engine_path), f"{engine_path} doesn't exists !"
        self.engine_path = engine_path

    def get_engine_eval_comment(self, fen: str) -> str:
        with chess.engine.SimpleEngine.popen_uci(self.engine_path) as engine:
            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(depth=30, time=2))
            score = info['score'].white()
            if isinstance(score, Cp):
                score = score.cp/100.0
        return f"[%eval {score}]"

    def get_engine_best_line(self, fen: str) -> str:
        line = ""
        with chess.engine.SimpleEngine.popen_uci(self.engine_path) as engine:
            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(depth=30, time=2))
            if len(info['pv']) > 0:
                if board.turn == chess.BLACK:
                    line += f"{board.fullmove_number}... "
                for move in info['pv']:
                    san = board.san(move)
                    if board.turn == chess.WHITE:
                        line += f"{board.fullmove_number}. {san} "
                    else:
                        line += f"{san} "
                    board.push(move)
        return line

    def analyze_game(self, game: chess.pgn.Game) -> chess.pgn.Game:
        with chess.engine.SimpleEngine.popen_uci(self.engine_path) as engine:
            board = game.board()
            node = game
            prev_score = PovScore(Cp(0), turn=chess.WHITE)
            for move in game.mainline_moves():
                san_move = board.san(move)
                board.push(move)
                node = node.variations[0]
                annotation = None

                info = engine.analyse(board, chess.engine.Limit(depth=30, time=2))
                crt_score = info['score']
                self.dbg.print(f"{math.ceil(board.ply()/2)} {san_move} (cp={crt_score.white()} ({type(crt_score.white())}), depth={info['depth']})", end=' ')

                if isinstance(prev_score.relative, Cp) and isinstance(crt_score.relative, Cp):
                    crt_score_cp = crt_score.pov(board.turn).score()
                    prev_score_cp = prev_score.pov(board.turn).score()
                    delta = Cp(crt_score_cp).wdl().expectation() - Cp(prev_score_cp).wdl().expectation()

                    if delta >= WinningChanceJudgements.BLUNDER.value:
                        self.dbg.print("BLUNDER")
                        annotation = NAG_BLUNDER
                    elif delta >= WinningChanceJudgements.MISTAKE.value:
                        self.dbg.print("MISTAKE")
                        annotation = NAG_MISTAKE
                    elif delta >= WinningChanceJudgements.INACCURACY.value:
                        self.dbg.print("INACCURACY")
                        annotation = NAG_DUBIOUS_MOVE
                    else:
                        self.dbg.print("")
                # Mate Created
                elif isinstance(prev_score.relative, Cp) and (isinstance(crt_score.relative, Mate) or isinstance(crt_score.relative, MateGiven)):
                    if prev_score.relative.score() < -999:
                        self.dbg.print("INACCURACY")
                        annotation = NAG_DUBIOUS_MOVE
                    elif prev_score.relative.score() < -700:
                        self.dbg.print("MISTAKE")
                        annotation = NAG_MISTAKE
                    else:
                        self.dbg.print("BLUNDER")
                        annotation = NAG_BLUNDER

                # Mate Lost
                elif (isinstance(prev_score.relative, Mate) or isinstance(prev_score.relative, MateGiven)) and isinstance(crt_score.relative, Cp):
                    if crt_score.relative.score() > 999:
                        self.dbg.print("INACCURACY")
                        annotation = NAG_DUBIOUS_MOVE
                    elif crt_score.relative.score() > 700:
                        self.dbg.print("MISTAKE")
                        annotation = NAG_MISTAKE
                    else:
                        self.dbg.print("BLUNDER")
                        annotation = NAG_BLUNDER
                else:
                    self.dbg.print("")

                if isinstance(crt_score.white(), Cp):
                    node.comment = f"[%eval {crt_score.white().score() / 100}] " + node.comment
                else:
                    node.comment = f"[%eval {crt_score.white()}] " + node.comment

                if annotation:
                    node.nags.add(annotation)
                prev_score = crt_score
        return game
