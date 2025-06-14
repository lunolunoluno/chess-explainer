import re
import os.path
import sys

import chess.pgn
import logging

from sympy.printing.pytorch import torch
from tqdm import tqdm
from transformers import pipeline

from modules.datautils import has_game_comments, pgn_to_id, get_all_pgn_files, get_all_comments_in_game
from modules.utils import Debug
from modules.gameanalyzer import GameAnalyzer


class Controller:

    def __init__(self):
        self.dbg = Debug()

        self.data_raw_path = os.getenv("DATA_RAW_PATH")
        assert os.path.exists(self.data_raw_path), f"{self.data_raw_path} doesn't exists !"
        self.data_analyzed_path = os.getenv("DATA_ANALYZED_PATH")
        assert os.path.exists(self.data_analyzed_path), f"{self.data_analyzed_path} doesn't exists !"

        # remove the non-critical logs in the terminal
        logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)

    def analyze_annotated_games(self)->None:
        """
        Will take all the pgn files in DATA_RAW_PATH and analyzed all the games.
        Once analyzed, each game is saved in a pgn file in DATA_ANALYZED_PATH
        """
        ga = GameAnalyzer()

        pgn_files = get_all_pgn_files()
        print(f"Number of .pgn files to analyze in {self.data_raw_path}: {len(pgn_files)}")

        for pgnfile in tqdm(pgn_files):
            with open(pgnfile, "r", encoding="utf-8", errors="replace") as pgn_file:
                self.dbg.print(f"\nAnalyzing {pgnfile}...")
                while True:
                    try:
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            # No more game in the file so exit the while loop
                            break
                        if len(game.errors) > 0:
                            raise Exception(f"error in {game.headers}\n"+"\n".join([str(e) for e in game.errors]))
                    except Exception as e:
                        print(f"\nError parsing a game in {pgnfile}: {e}", file=sys.stderr)
                        continue # skip to the next game in pgnfile

                    str_exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=True)
                    pgn_str = game.accept(str_exporter)
                    if has_game_comments(pgn_str):
                        self.dbg.print(f"Analyzing {game.headers}...")
                        game_id = pgn_to_id(pgn_str)
                        analyzed_game_path = os.path.join(self.data_analyzed_path, f"{game_id}.pgn")
                        if not os.path.exists(analyzed_game_path):
                            ga.analyze_game(game)
                            with open(analyzed_game_path, "w", encoding="utf-8") as new_pgn:
                                exporter = chess.pgn.FileExporter(new_pgn)
                                game.accept(exporter)
                            self.dbg.print(f"\tGame saved as {analyzed_game_path}")
                        else:
                            self.dbg.print(f"\tGame already saved as {analyzed_game_path}")
                    else:
                        self.dbg.print(f"No comments in {pgnfile} -> {game.headers}")

    def save_good_comments_from_games(self)->None:
        """
        Will take all the pgn files in DATA_RAW_PATH and extract the comments that explain any player's mistake.
        Once analyzed, each game is saved in a pgn file in DATA_ANALYZED_PATH
        All the comments from a game will be saved in a csv file in DATA_COMMENTS_PATH
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pgn_files = get_all_pgn_files()
        print(f"Number of .pgn files to analyze in {self.data_raw_path}: {len(pgn_files)}")

        for pgnfile in tqdm(pgn_files):
            # extract all the comments from the pgn file
            str_exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=True)
            pgn = open(pgnfile)
            game = chess.pgn.read_game(pgn)
            while game is not None:
                comments = get_all_comments_in_game(game.accept(str_exporter))

                # remove comments with just the engine evaluation
                filtered_comments = [
                    comment for comment in comments
                    if re.sub(r'\s*\[%eval [^]]+\]\s*', '', comment).strip() != ''
                ]

                pipe = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct", device=device)
                for c in filtered_comments:
                    self.dbg.print(c)
                    messages = [
                        {"role": "system", "content": """You're job is to evaluate whether a chess annotation comment on the quality of the previously played move.
                                When Writing your answer, it is VERY IMPORTANT that you write RES followed by 1 if you think this chess annotation comment on the quality of the previously played move or a 0 if you think it doesn't.
                                After that, write a single sentence explaining your reasoning.
                                Don't write anything else after."""},
                        {"role": "user", "content": f"""Here is a comment made after a chess move evaluated as a mistake:
                                "{c}"
                                Do you think that this comment explains the mistake made by the player ?
                                Write RES: 1 if yes and write RES: 0 if no.
                                After that, write a single sentence explaining your reasoning.
                                Don't write anything else after."""},
                    ]
                    while True:
                        output = pipe(messages)
                        last_assistant_message = next(msg for msg in reversed(output[0]['generated_text']) if msg["role"] == "assistant")
                        output_txt = last_assistant_message['content'].strip()
                        if "RES:" in output_txt:
                            try:
                                res = output_txt.split("RES:")[1].strip()[0]
                                if res in ("0", "1"):
                                    self.dbg.print(f"RES = {res}")
                                    break  # Exit the loop on valid result
                            except Exception:
                                pass  # Malformed after RES:

                        self.dbg.print("Invalid or incomplete response. Re-asking the question...")
                    self.dbg.print("Is comment good ? ", res)
                game = chess.pgn.read_game(pgn)
            pgn.close()


