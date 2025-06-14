import os
import re
import chess.pgn
import torch
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import InferenceClient

from controller import Controller
from modules.gameanalyzer import GameAnalyzer
from modules.utils import Debug
from modules.datautils import get_all_comments_after_error, get_all_comments_in_game

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    dbg = Debug(debug=True)
    dbg.print("Debug: On")

    # ctrl = Controller()
    # ctrl.analyze_annotated_games()

    filepath = os.path.join(".", "data", "analyzed", "Aa0p17YmJW9A.pgn")

    str_exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=True)
    pgn = open(filepath)
    game = chess.pgn.read_game(pgn)
    comments = get_all_comments_in_game(game.accept(str_exporter))
    pgn.close()

    # remove comments with just the engine evaluation
    filtered_comments = [
        comment for comment in comments
        if re.sub(r'\s*\[%eval [^]]+\]\s*', '', comment).strip() != ''
    ]
    print(f"{len(filtered_comments)} comments found !")

#     pipe = pipeline("text-generation", model="microsoft/phi-2", device=device)
#     prompt = """
# <|system|>
# You are an AI whose job is to evaluate whether a chess annotation comment on the quality of the previously played move.
# When Writing your answer, it is VERY IMPORTANT that you write RES followed by 1 if you think this chess annotation comment on the quality of the previously played move or a 0 if you think it doesn't.
# <|user|>
# Here is a comment made after a chess move evaluated as a mistake:
# "By playing this move, white allows black to capture the bishop on e2 for free."
# Do you think that this comment explains the mistake made by the player ?
# Write RES: 1 if yes and write RES: 0 if no.
# <|assistant|>
#     """
#     output = pipe(prompt, max_new_tokens=50)
#
#     print(output)
#     print(output[0]['generated_text'].split('<|assistant|>')[1])

    pipe = pipeline("text-generation", model="microsoft/phi-2", device=device)
    for c in filtered_comments:
        print(c)
        prompt = f"""
<|system|>
You are an AI whose job is to evaluate whether a chess annotation comment on the quality of the previously played move.
When Writing your answer, it is VERY IMPORTANT that you write RES followed by 1 if you think this chess annotation comment on the quality of the previously played move or a 0 if you think it doesn't.
After that, write a single sentence explaining your reasoning.
Don't write anything else after.
<|user|>
Here is a comment made after a chess move evaluated as a mistake:
"{c}"
Do you think that this comment explains the mistake made by the player ?
Write RES: 1 if yes and write RES: 0 if no.
After that, write a single sentence explaining your reasoning.
Don't write anything else after.
<|assistant|>
        """
        output = pipe(prompt)
        output_txt = output[0]['generated_text'].split('<|assistant|>')[1].strip()
        print(output_txt)
        print(40 * "=")
        # messages.append({
        #     "role":"assistant",
        #     "content":response
        # })
