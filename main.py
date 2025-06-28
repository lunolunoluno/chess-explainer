import os

import chess.pgn

from controller import Controller
from modules.utils import Debug, LLM

if __name__ == "__main__":
    dbg = Debug(debug=True)
    dbg.print("Debug: On")

    llm = LLM()
    llm.set_model_id("microsoft/Phi-3-mini-4k-instruct")

    ctrl = Controller()
    ctrl.save_good_comments_from_games()
    ctrl.reformulate_good_comments()

