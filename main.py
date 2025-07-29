import os

import chess.pgn
import pandas as pd

from controller import Controller
from modules.utils import Debug, LLM
from modules.evaluator import Evaluator

import sacrebleu
from sacrebleu import sentence_bleu
from transformers import pipeline

if __name__ == "__main__":
    # dbg = Debug(debug=True)
    # dbg.print("Debug: On")
    #
    # ctrl = Controller()
    # # ctrl.save_good_comments_from_games()
    # # ctrl.reformulate_good_comments()
    #
    # dataset_path = os.path.join(".", "data", "filtered_merged_20250708_134716.csv")
    # ctrl.train_model(dataset_path, ["context", "moves"], "reformulated")
    #
    #
    # df = pd.read_csv(dataset_path)
    # num_of_rows = df.shape[0]
    # print("Number of data :", num_of_rows)

    # # Use a pipeline as a high-level helper
    # from transformers import pipeline
    #
    # pipe = pipeline("text-generation", model="google/gemma-3-1b-it")
    # messages = [
    #     {"role": "user", "content": "Translate to English: 'Je suis étudiant."},
    # ]
    # answer = pipe(messages)
    # generated = answer[0]['generated_text'][1]['content']
    #
    # # Optional: Print generated output for inspection
    # print("Generated:", generated)
    #
    # # Step 4: Extract only the generated part (you might need to clean this better depending on the model output)
    # # For this example, let's assume we manually extract the translation part:
    # candidate = generated
    #
    # # Step 5: Define the reference(s)
    # references = ["I am a student"]
    #
    # # Step 6: Compute BLEU using sacrebleu
    # bleu = sacrebleu.corpus_bleu([candidate], [references])
    # print(bleu)
    # print(f"BLEU score: {bleu.score:.2f}")
    ev = Evaluator()

    candidates = [
        "This one is the same",
        "This one is shorter",
        "Longer",
        "This one is a bit short"
    ]
    references = [
        "This one is the same",
        "Shorter",
        "This one is longer",
        "This one is a bit shorter"
    ]

    for hyp, ref in zip(candidates, references):
        bleu = ev.bleu_evaluation(ref, hyp)
        rouge1, rouge2, rougeL = ev.rouge_evaluation(ref, hyp)
        meteor = ev.meteor_evaluation(ref, hyp)
        bertscore = ev.bertscore_evaluation(ref, hyp)
        print(f"""
        Reference: {ref}
        Hypothesis: {hyp}
        
        BLEU: {bleu}
        ROUGE-1: {rouge1*100}
        ROUGE-2: {rouge2*100}
        ROUGE-L: {rougeL*100}
        METEOR: {meteor*100}
        BERTScore: {bertscore*100}
        """+"="*20)

