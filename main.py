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
    dbg = Debug(debug=True)
    dbg.print("Debug: On")

    ctrl = Controller()
    ctrl.analyze_annotated_games()

    # filepath = os.path.join(".", "data", "analyzed", "Aa0p17YmJW9A.pgn")
    #
    # str_exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=True)
    # pgn = open(filepath)
    # game = chess.pgn.read_game(pgn)
    # comments = get_all_comments_in_game(game.accept(str_exporter))
    # pgn.close()
    #
    # # remove comments with just the engine evaluation
    # filtered_comments = [
    #     comment for comment in comments
    #     if re.sub(r'\s*\[%eval [^]]+\]\s*', '', comment).strip() != ''
    # ]
    # print(f"{len(filtered_comments)} comments found !")
    #
    # pipe = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct", device=device)
    # for c in filtered_comments:
    #     print(c)
    #     messages = [
    #         {"role": "system", "content": """You're job is to evaluate whether a chess annotation comment on the quality of the previously played move.
    #             When Writing your answer, it is VERY IMPORTANT that you write RES followed by 1 if you think this chess annotation comment on the quality of the previously played move or a 0 if you think it doesn't.
    #             After that, write a single sentence explaining your reasoning.
    #             Don't write anything else after."""},
    #         {"role": "user", "content": f"""Here is a comment made after a chess move evaluated as a mistake:
    #             "{c}"
    #             Do you think that this comment explains the mistake made by the player ?
    #             Write RES: 1 if yes and write RES: 0 if no.
    #             After that, write a single sentence explaining your reasoning.
    #             Don't write anything else after."""},
    #     ]
    #     while True:
    #         output = pipe(messages)
    #         print(output)
    #         last_assistant_message = next(msg for msg in reversed(output[0]['generated_text']) if msg["role"] == "assistant")
    #         output_txt = last_assistant_message['content'].strip()
    #         if "RES:" in output_txt:
    #             try:
    #                 res = output_txt.split("RES:")[1].strip()[0]
    #                 if res in ("0", "1"):
    #                     print(f"RES = {res}")
    #                     break  # Exit the loop on valid result
    #             except Exception:
    #                 pass  # Malformed after RES:
    #
    #         print("Invalid or incomplete response. Re-asking the question...")
    #     print("Is comment good ? ", res)
    #     print(40 * "=")
