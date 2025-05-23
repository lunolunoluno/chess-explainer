import re
from typing import List

def assert_pgn_well_formatted(pgn: str)->None:
    # Work in progress
    pass

def has_game_comments(pgn: str)->bool:
    self.assert_pgn_well_formatted(pgn)
    comment_pattern = r'\{.*?\}'
    return re.search(comment_pattern, pgn) is not None

def get_all_comments_after_error(pgn: str)->List[str]:
    assert_pgn_well_formatted(pgn)
    pgn_cleaned = re.sub(r'[\r\n]+', '', pgn) # remove all return characters
    pattern = r'\$\d[ \t]*\{(.*?)}' # get all comments after $ followed by a number (this is because python-chess replace the NAGs with those for some reason)
    return re.findall(pattern, pgn_cleaned)

