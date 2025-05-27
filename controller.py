import os.path
import sys

import chess.pgn
import logging

from tqdm import tqdm

from modules.datautils import has_game_comments, pgn_to_id
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
        Once the game(s) is/are analyzed in a raw pgn file with no error occurring, the file is deleted.
        """
        ga = GameAnalyzer()

        pgn_files = [
            os.path.join(self.data_raw_path, file)
            for file in os.listdir(self.data_raw_path)
            if os.path.isfile(os.path.join(self.data_raw_path, file)) and file.lower().endswith(".pgn") # for debug purposes only
        ]
        print(f"Number of .pgn files to analyze in {self.data_raw_path}: {len(pgn_files)}")

        for pgnfile in tqdm(pgn_files):
            with open(pgnfile, "r", encoding="utf-8", errors="replace") as pgn_file:
                self.dbg.print(f"\nAnalyzing {pgnfile}...")
                while True:
                    try:
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            # No more game so exit the while loop
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






