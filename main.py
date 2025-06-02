import os

import chess
import torch
from transformers import pipeline

from controller import Controller
from modules.gameanalyzer import GameAnalyzer
from modules.utils import Debug

if __name__ == "__main__":
    dbg = Debug(debug=True)
    dbg.print("Debug: On")

    # ctrl = Controller()
    # ctrl.analyze_annotated_games()

    pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

    # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot",
        },
        {"role": "user", "content": "What do you know about chess?"},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    print(outputs[0]["generated_text"])