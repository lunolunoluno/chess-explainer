import io
import os
import re
import base64
import hashlib
from typing import List

import chess.pgn
import pandas as pd
from transformers import Pipeline
from datasets import Dataset

from modules.gameanalyzer import GameAnalyzer
from modules.utils import Debug


def remove_return_char(pgn_str: str) -> str:
    return re.sub(r'[\r\n]+', '', pgn_str)  # remove all return characters


def has_game_comments(pgn_str: str) -> bool:
    pgn_cleaned = remove_return_char(pgn_str)
    comment_pattern = r'\{.*?\}'
    return re.search(comment_pattern, pgn_cleaned) is not None


def get_all_comments_after_error(pgn_str: str) -> List[str]:
    pgn_cleaned = remove_return_char(pgn_str)
    pattern = r'\$\d[ \t]*\{(.*?)}'  # get all comments after $ followed by a number (this is because python-chess replace the NAGs with those for some reason)
    return re.findall(pattern, pgn_cleaned)


def get_all_comments_in_game(pgn_str: str) -> List[str]:
    pgn_cleaned = remove_return_char(pgn_str)
    pattern = r'\{(.*?)}'
    return re.findall(pattern, pgn_cleaned)


def get_all_comments_and_lines_in_game(game: chess.pgn.Game, initial_context: str) -> List[dict]:
    ga = GameAnalyzer()
    node = game
    board = chess.Board()
    partial_game = []
    res = []
    # Games with custom fen start are not supported yet
    if 'FEN' not in game.headers:
        while node.variations:
            next_node = node.variations[0]
            move = next_node.move
            fen_before = board.fen()
            san = board.san(move)
            board.push(move)
            fen_after = board.fen()
            if board.turn == chess.BLACK:
                partial_game.append(f"{board.fullmove_number}.")
            partial_game.append(san)

            comment = next_node.comment.strip()

            # Remove pure eval-only comments
            cleaned_comment = re.sub(r'\s*\[%eval [^]]+\]\s*', '', comment).strip()
            # Remove newline, tab, carriage return
            cleaned_comment = re.sub(r'[\n\t\r]+', ' ', cleaned_comment)
            # Remove double spaces
            cleaned_comment = re.sub(r" +", " ", cleaned_comment)

            if cleaned_comment != '':
                moves = ' '.join(partial_game)
                res.append({
                    "comment": cleaned_comment,
                    "moves": moves,
                    "context": initial_context + ". Last move played: " + get_last_move_from_line_as_string(moves, game.headers),
                    "engine_eval": ga.get_engine_eval_comment(fen_after),
                    "engine_best_line": ga.get_engine_best_line(fen_after),
                    "engine_best_alternative": ga.get_engine_best_line(fen_before)
                })
            node = next_node
    return res


def pgn_to_id(pgn_str: str):
    hash_bytes = hashlib.sha256(pgn_str.encode()).digest()
    return base64.urlsafe_b64encode(hash_bytes)[:12].decode()


def get_all_pgn_files() -> List[str]:
    data_raw_path = os.getenv("DATA_RAW_PATH")
    return [
        os.path.join(data_raw_path, file)
        for file in os.listdir(data_raw_path)
        if os.path.isfile(os.path.join(data_raw_path, file)) and file.lower().endswith(".pgn")
    ]


def filter_good_comments(pipe: Pipeline, comments: List[dict]) -> List[dict]:
    prompt_model = lambda c: [
        {"role": "system", "content": """You're job is to evaluate whether a chess annotation comment on the quality of the previously played move.
                                    When Writing your answer, it is VERY IMPORTANT that you write RES followed by 1 if you think this chess annotation comment on the quality of the previously played move or a 0 if you think it doesn't.
                                    After that, write a single sentence explaining your reasoning.
                                    Don't write anything else after."""},
        {"role": "user", "content": f"""Here is a comment made after a chess move evaluated as a mistake:
                                    Comment: "{c}"
                                    Do you think that this comment explains the mistake made by a player ?
                                    Write RES: 1 if yes and write RES: 0 if no.
                                    After that, write a single sentence explaining your reasoning.
                                    Don't write anything else after."""},
    ]
    prompts = [prompt_model(comment['comment']) for comment in comments]

    def __filter_good_comments__(p: Pipeline, pr: list, depth: int) -> List[dict]:
        # TODO: Make batch_size a parameter that can be changed depending on the machine on which the code is run
        outputs = p(pr, batch_size=4)
        bad_outputs = []
        good_comments = []
        for comment, o in zip(comments, outputs):
            last_assistant_message = next(msg for msg in reversed(o[0]['generated_text']) if msg["role"] == "assistant")
            output_txt = last_assistant_message['content'].strip()
            if "RES:" in output_txt:
                after_res = output_txt.split("RES:")[1].strip()
                res = after_res[0]
                if res in ("0", "1"):
                    good_comments.append({
                        "comment": comment['comment'],
                        "good": res == "1",
                        "reasoning": after_res[1:],
                        "moves": comment['moves'],
                        "context": comment['context'],
                        "engine_eval": comment['engine_eval'],
                        "engine_best_line": comment['engine_best_line'],
                        "engine_best_alternative": comment['engine_best_alternative']
                    })
                else:
                    bad_outputs.append(o[0]['generated_text'][:-1])
            else:
                bad_outputs.append(o[0]['generated_text'][:-1])

        if len(bad_outputs) > 0 and depth < 5:
            good_comments += __filter_good_comments__(p, bad_outputs, depth + 1)

        return good_comments

    return __filter_good_comments__(pipe, prompts, 0)


def is_comment_explaining_mistake(pipe: Pipeline, comment: str) -> bool:
    dbg = Debug()
    dbg.print("analyzing ->", comment)
    err = True
    while err:
        messages = [
            {"role": "system", "content": """You're job is to evaluate whether a chess annotation comment on the quality of the previously played move.
                                        When Writing your answer, it is VERY IMPORTANT that you write RES followed by 1 if you think this chess annotation comment on the quality of the previously played move or a 0 if you think it doesn't.
                                        After that, write a single sentence explaining your reasoning.
                                        Don't write anything else after."""},
            {"role": "user", "content": f"""Here is a comment made after a chess move evaluated as a mistake:
                                        "{comment}"
                                        Do you think that this comment explains the mistake made by the player ?
                                        Write RES: 1 if yes and write RES: 0 if no.
                                        After that, write a single sentence explaining your reasoning.
                                        Don't write anything else after."""},
        ]
        while True:
            output = pipe(messages)
            err = False
            last_assistant_message = next(
                msg for msg in reversed(output[0]['generated_text']) if msg["role"] == "assistant")
            output_txt = last_assistant_message['content'].strip()
            if "RES:" in output_txt:
                try:
                    res = output_txt.split("RES:")[1].strip()[0]
                    if res in ("0", "1"):
                        dbg.print("Is comment good ?", output_txt)
                        break  # Exit the loop on valid result
                except Exception:
                    pass  # Malformed after RES:

            dbg.print("Invalid or incomplete response. Re-asking the question...")
    return True if res == "1" else False


def get_last_move_from_line_as_string(pgn_line: str, header: chess.pgn.Headers) -> str:
    pgn = io.StringIO(pgn_line)
    game = chess.pgn.read_game(pgn)
    if len(game.errors) == 0:
        piece_names = {
            chess.PAWN: "Pawn",
            chess.KNIGHT: "Knight",
            chess.BISHOP: "Bishop",
            chess.ROOK: "Rook",
            chess.QUEEN: "Queen",
            chess.KING: "King"
        }
        last_move = game.end()
        color = 'Black' if last_move.turn() else 'White'

        move = last_move.move
        board = last_move.board()
        board.pop()

        if board.is_kingside_castling(move):
            move_str = "kingside castle"
        elif board.is_queenside_castling(move):
            move_str = "queenside castle"
        else:
            piece_name = piece_names[board.piece_at(move.from_square).piece_type]
            move_str = f"{piece_name} {'takes on' if board.is_capture(move) else 'to'} {chess.square_name(move.to_square)}{' promote to a ' + piece_names[move.promotion] if move.promotion else ''}"

        return f"{color} plays {move_str}{' with check' if board.gives_check(move) else ''}"
    else:
        errors = '\n\n'.join(str(e) for e in game.errors)
        raise Exception(f"The following error(s) were encountered during the parsing of the pgn {header} :\n{errors}")


def create_dataset(df_dataset: pd.DataFrame, inputs_columns: List[str], label_column: str) -> Dataset:
    dbg = Debug()

    # Create hugging face dataset
    def prompt_model(row) -> str:
        info = "\n".join([f"{col}: {row[col]}" for col in inputs_columns])
        return re.sub(r'\t| {2,}', '', f"""
                        Based on the following information: 
                        {info}
                        Generate a comment explaining the error that the player just made
                        Comment: \n""".strip())

    dbg.print("Creating dataset...")
    df_dataset["prompt"] = df_dataset.apply(prompt_model, axis=1)
    df_dataset["full_text"] = df_dataset["prompt"] + " " + df_dataset[label_column].astype(str)

    # Create dataset with only necessary columns
    dataset = Dataset.from_pandas(df_dataset[["prompt", label_column, "full_text"]])
    return dataset

def safe_folder_name(name: str, replace_with: str = "_") -> str:
    safe_name = re.sub(r'[<>:"/\\|?*\']', replace_with, name)
    safe_name = safe_name.strip(" .")
    return safe_name
