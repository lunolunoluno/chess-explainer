import os
import re
import base64
import hashlib
from typing import List

import chess.pgn
import torch
from transformers import Pipeline

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


def get_all_comments_and_lines_in_game(game: chess.pgn.Game) -> List[dict]:
    node = game
    board = chess.Board()
    partial_game = []
    res = []
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
            res.append({
                "moves": ' '.join(partial_game),
                "comment": cleaned_comment
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
                                    Do you think that this comment explains the mistake made by the player ?
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
                res = output_txt.split("RES:")[1].strip()[0]
                if res in ("0", "1"):
                    good_comments.append({
                        "moves": comment['moves'],
                        "comment": comment['comment'],
                        "good": res == "1"
                    })
                else:
                    bad_outputs.append(o[0]['generated_text'][:-1])
            else:
                bad_outputs.append(o[0]['generated_text'][:-1])

        if len(bad_outputs) > 0 and depth < 5:
            good_comments += __filter_good_comments__(p, bad_outputs, depth + 1)

        return good_comments

    dbg = Debug()
    dbg.print(f"Analyzing {len(comments)} comments...")
    return __filter_good_comments__(pipe, prompts, 0)


def is_comment_explaining_mistake(pipe: Pipeline, comment: str) -> bool:
    dbg = Debug()
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
