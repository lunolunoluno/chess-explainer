import os
import re
import base64
import hashlib
from typing import List

def remove_return_char(pgn_str: str)->str:
    return re.sub(r'[\r\n]+', '', pgn_str)  # remove all return characters

def has_game_comments(pgn_str: str)->bool:
    pgn_cleaned = remove_return_char(pgn_str)
    comment_pattern = r'\{.*?\}'
    return re.search(comment_pattern, pgn_cleaned) is not None

def get_all_comments_after_error(pgn_str: str)->List[str]:
    pgn_cleaned = remove_return_char(pgn_str)
    pattern = r'\$\d[ \t]*\{(.*?)}' # get all comments after $ followed by a number (this is because python-chess replace the NAGs with those for some reason)
    return re.findall(pattern, pgn_cleaned)

def get_all_comments_in_game(pgn_str: str)->List[str]:
    pgn_cleaned = remove_return_char(pgn_str)
    pattern = r'\{(.*?)}'
    return re.findall(pattern, pgn_cleaned)


def pgn_to_id(pgn_str: str):
    hash_bytes = hashlib.sha256(pgn_str.encode()).digest()
    return base64.urlsafe_b64encode(hash_bytes)[:12].decode()

def get_all_pgn_files()->List[str]:
    data_raw_path = os.getenv("DATA_RAW_PATH")
    return [
        os.path.join(data_raw_path, file)
        for file in os.listdir(data_raw_path)
        if os.path.isfile(os.path.join(data_raw_path, file)) and file.lower().endswith(".pgn")
    ]
