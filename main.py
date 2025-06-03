import os
import re
import chess.pgn
import torch
from tqdm import tqdm
from transformers import pipeline
from huggingface_hub import InferenceClient

from controller import Controller
from modules.gameanalyzer import GameAnalyzer
from modules.utils import Debug
from modules.datautils import get_all_comments_after_error, get_all_comments_in_game

if __name__ == "__main__":
    dbg = Debug(debug=True)
    dbg.print("Debug: On")

    # ctrl = Controller()
    # ctrl.analyze_annotated_games()

    filepath = os.path.join(".", "data", "analyzed", "Aa0p17YmJW9A.pgn")

    str_exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=True)
    pgn = open(filepath)
    game = chess.pgn.read_game(pgn)
    comments = get_all_comments_after_error(game.accept(str_exporter))
    pgn.close()

    # remove comments with just the engine evaluation
    filtered_comments = [
        comment for comment in comments
        if re.sub(r'\s*\[%eval [^]]+\]\s*', '', comment).strip() != ''
    ]
    print(f"{len(filtered_comments)} comments found !")

    api_key = os.getenv("HUGGING_FACE_TOKEN")
    client = InferenceClient(
        provider="hf-inference",
        api_key=api_key,
    )

    # completion = client.chat.completions.create(
    #     model="microsoft/Phi-3-mini-4k-instruct",
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": "What is the capital of France?"
    #         }
    #     ],
    # )
    # print(completion.choices[0].message)

    messages = [
        {
            "role": "system",
            "content": "Your job is to analyze how relevant a comment is about a chess game. The user will give you some comments found in games and you'll indicate whether the comment explains the mistake made by a player.",
        }
    ]

    for c in filtered_comments:
        print(c)
        messages.append({
            "role":"user",
            "content":f"""
            Here is a comment made after a chess move evaluated as a mistake:

            {c}

            Do you think that this comment explains the mistake made by the player ?
            """
        })
        completion = client.chat.completions.create(
            model="microsoft/Phi-3-mini-4k-instruct",
            messages=messages
        )
        response = completion.choices[0].message.content
        print(response)
        print(40 * "=")
        messages.append({
            "role":"assistant",
            "content":response
        })