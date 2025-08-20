import os
import io

import chess.pgn

from chess.engine import PovScore, Cp, Mate, MateGiven

from modules.gameanalyzer import GameAnalyzer

# These 2 lines prevent pytorch from trying to use Triton
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

import re
from controller import Controller
from modules.utils import Debug, LLM


if __name__ == "__main__":
    # pgn_str = "1. e4 e5 2. Nf3 Nf6 3. Nxe5 Nc6"
    # pgn = io.StringIO(pgn_str)
    # game = chess.pgn.read_game(pgn)
    # fen = game.end().board().fen()
    # ga = GameAnalyzer()
    # eval = ga.get_engine_eval_comment(fen)
    # print(eval)

    dbg = Debug(debug=True)
    dbg.print("Debug: On")
    llm = LLM()

    ctrl = Controller()
    dbg.print("\nSAVE GOOD COMMENTS FROM GAMES\n"+"="*40)
    ctrl.save_good_comments_from_games()
    # dbg.print("\nREFORMULATE COMMENTS\n"+"="*40)
    # ctrl.reformulate_good_comments()
    # dbg.print("\nSAVE COMMENTS AS A SINGLE CSV\n"+"="*40)
    # dataset_path = ctrl.save_comments_as_csv()
    #
    # checkpoint_name = ctrl.train_model(dataset_path, ["moves", "engine_eval", "engine_best_line", "engine_best_alternative"], "reformulated")
    # model, tokenizer = ctrl.load_model_and_tokenizer_from_checkpoint(checkpoint_name)
    #
    # prompt = re.sub(r'\t| {2,}', '', f"""
    #                     Based on the following information:
    #                     moves: 1. e4 e5 2. Nf3 Nf6 3. Nxe5 Nc6 4. Nxc6 dxc6 5. Nc3 Bc5 6. d3
    #                     engine_eval: [%eval -0.93]
    #                     engine_best_line: 6...Ng4 7. Be3 Nxe3 8. fxe3 Bxe3
    #                     engine_best_alternative: 6. h3 O-O 7. d3 b5 8. Be2
    #                     Generate a comment explaining the error that the player just made
    #                     Comment: \n""".strip())
    #
    # comment = ctrl.prompt_model(model, tokenizer, prompt)
    #
    # # The comment should say something like White should have played 6. h3 preventing Black from playing Ng4
    # print(comment)
    #

