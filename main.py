import os
import sys
import chess.pgn
import pandas as pd
from tqdm import tqdm
from modules.datautils import get_all_pgn_files, has_game_comments, get_all_comments_and_lines_in_game
from modules.utils import Debug

# These 2 lines prevent pytorch from trying to use Triton
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'



if __name__ == "__main__":

    dbg = Debug(debug=True)
    dbg.print("Debug: On")

    pgn_files = get_all_pgn_files()
    all_comments = []

    for pgnfile in tqdm(pgn_files):
        with open(pgnfile, encoding="utf-8", errors="replace") as pgn_game:
            game = chess.pgn.read_game(pgn_game)
            while game is not None:
                header = game.headers
                if 'White' in header and header['White'].strip() != '':
                    context = f"This is a game between {header['White']} (as White)"
                else:
                    context = "This is a game between an unknown player (as White)"
                if 'Black' in header and header['Black'].strip() != '':
                    context += f" and {header['Black']} (as Black)"
                else:
                    context += " and an unknown player (as Black)"
                str_exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=True)
                pgn_str = game.accept(str_exporter)
                comments = get_all_comments_and_lines_in_game(game, context)
                all_comments += comments
                game = chess.pgn.read_game(pgn_game)

    df = pd.DataFrame(all_comments)
    df.to_csv(os.path.join('.', 'data', 'raw_data.csv'), index=False)