import re
import os.path
import sys

import chess.pgn
import logging
import pandas as pd

from sympy.printing.pytorch import torch
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from modules.datautils import has_game_comments, pgn_to_id, get_all_pgn_files, is_comment_explaining_mistake, \
    get_all_comments_and_lines_in_game, filter_good_comments
from modules.utils import Debug
from modules.gameanalyzer import GameAnalyzer


class Controller:

    def __init__(self):
        self.dbg = Debug()

        self.data_raw_path = os.getenv("DATA_RAW_PATH")
        assert os.path.exists(self.data_raw_path), f"{self.data_raw_path} doesn't exists !"
        self.data_analyzed_path = os.getenv("DATA_ANALYZED_PATH")
        assert os.path.exists(self.data_analyzed_path), f"{self.data_analyzed_path} doesn't exists !"
        self.data_commented_path = os.getenv("DATA_COMMENTS_PATH")
        assert os.path.exists(self.data_commented_path), f"{self.data_commented_path} doesn't exists !"

        # remove the non-critical logs in the terminal
        logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)

    def analyze_annotated_games(self) -> None:
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
                            raise Exception(f"error in {game.headers}\n" + "\n".join([str(e) for e in game.errors]))
                    except Exception as e:
                        print(f"\nError parsing a game in {pgnfile}: {e}", file=sys.stderr)
                        continue  # skip to the next game in pgnfile

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

    def save_good_comments_from_games(self) -> None:
        """
        Will take all the pgn files in DATA_RAW_PATH and extract the comments that explain any player's mistake.
        Once analyzed, each game is saved in a pgn file in DATA_ANALYZED_PATH
        All the comments from a game will be saved in a csv file in DATA_COMMENTS_PATH
        """
        # Create the pipeline that will be used to evaluate the comments
        model_id = "microsoft/Phi-3-mini-4k-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        # get all the pgn files that will be evaluated
        pgn_files = get_all_pgn_files()
        self.dbg.print(f"Number of .pgn files to analyze in {self.data_raw_path}: {len(pgn_files)}")

        for pgnfile in tqdm(pgn_files):
            self.dbg.print(f"Analyzing {pgnfile}...")

            # Read game in pgn file
            pgn_game = open(pgnfile)
            game = chess.pgn.read_game(pgn_game)
            while game is not None:
                str_exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=True)
                pgn_str = game.accept(str_exporter)
                cvs_path = os.path.join(self.data_commented_path, f"{pgn_to_id(pgn_str)}.csv")
                if not os.path.exists(cvs_path):
                    comments = get_all_comments_and_lines_in_game(game)
                    if len(comments) > 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                        good_comments = filter_good_comments(pipe, comments)

                        df = pd.DataFrame(good_comments)
                        df.to_csv(cvs_path, index=False)

                game = chess.pgn.read_game(pgn_game)

    def save_good_comments_from_games_old(self) -> None:
        """
        Will take all the pgn files in DATA_RAW_PATH and extract the comments that explain any player's mistake.
        Once analyzed, each game is saved in a pgn file in DATA_ANALYZED_PATH
        All the comments from a game will be saved in a csv file in DATA_COMMENTS_PATH
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        pgn_files = get_all_pgn_files()
        print(f"Number of .pgn files to analyze in {self.data_raw_path}: {len(pgn_files)}")

        pipe = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct", device=device)

        for pgnfile in tqdm(pgn_files):
            self.dbg.print(f"analyzing {pgnfile}")

            # extract all the comments from the pgn file
            pgn_game = open(pgnfile)
            game = chess.pgn.read_game(pgn_game)
            while game is not None:
                str_exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=True)
                pgn_str = game.accept(str_exporter)
                cvs_path = os.path.join(self.data_commented_path, f"{pgn_to_id(pgn_str)}.csv")
                if not os.path.exists(cvs_path):
                    node = game
                    board = chess.Board()
                    partial_game = []
                    csv_rows = []
                    while node.variations:
                        next_node = node.variations[0]
                        move = next_node.move
                        san = board.san(move)
                        board.push(move)
                        if board.turn == chess.BLACK:
                            partial_game.append(f"{board.fullmove_number}.")
                        partial_game.append(san)

                        comment = next_node.comment.strip()

                        # Remove pure eval-only comments
                        cleaned_comment = re.sub(r'\s*\[%eval [^]]+\]\s*', '', comment).strip()
                        if cleaned_comment != '':
                            self.dbg.print("Moves leading to the comment:", ' '.join(partial_game))
                            good_comment = is_comment_explaining_mistake(pipe, cleaned_comment)
                            if good_comment:
                                csv_rows.append({
                                    "Moves": ' '.join(partial_game),
                                    "Comment": cleaned_comment
                                })
                        node = next_node
                    df = pd.DataFrame(csv_rows)
                    df.to_csv(cvs_path, index=False)
                game = chess.pgn.read_game(pgn_game)

            pgn_game.close()
