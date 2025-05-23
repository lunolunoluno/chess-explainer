import re
from typing import List

def has_game_comments(pgn: str)->bool:
    comment_pattern = r'\{.*?\}'
    return re.search(comment_pattern, pgn) is not None

def get_all_comments_after_error(pgn: str)->List[str]:
    pgn_cleaned = re.sub(r'[\r\n]+', '', pgn) # remove all return characters
    pattern = r'\$\d[ \t]*\{(.*?)}' # get all comments after $ followed by a number (this is because python-chess replace the NAGs with those for some reason)
    return re.findall(pattern, pgn_cleaned)

