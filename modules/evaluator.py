import nltk
import sacrebleu
import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from tqdm import tqdm
tqdm.pandas()

from modules.datautils import remove_san_from_text, safe_folder_name



class Evaluator:
    def __init__(self):
        nltk.download('wordnet')
        nltk.download('omw-1.4')

    # BLEU: a Method for Automatic Evaluation of Machine Translation
    def bleu_evaluation(self, reference: str, hypothesis: str) -> float:
        bleu = sacrebleu.corpus_bleu([hypothesis], [[reference]])
        return bleu.score

    # Rouge: A package for automatic evaluation of summaries
    def rouge_evaluation(self, reference: str, hypothesis: str) -> (float, float, float):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return scores['rouge1'].precision, scores['rouge2'].precision, scores['rougeL'].precision

    # METEOR: An automatic metric for MT evaluation with improved correlation with human judgments
    def meteor_evaluation(self, reference: str, hypothesis: str) -> float:
        # Tokenize both reference and hypothesis
        reference_tokens = reference.strip().split()
        hypothesis_tokens = hypothesis.strip().split()

        # Pass a list of tokenized references and a tokenized hypothesis
        return meteor_score([reference_tokens], hypothesis_tokens)

    # BERTSCORE: EVALUATING TEXT GENERATION WITH BERT
    def bertscore_evaluation(self, reference: str, hypothesis: str) -> float:
        # Inputs must be lists of strings
        P, R, F1 = bert_score([hypothesis], [reference], lang="en", rescale_with_baseline=False)
        return F1.item()  # Return the scalar F1 score

    def create_model_graphs(self, df_results: pd.DataFrame, save_eval: bool, show_graphs: bool, folder_name: str):
        def __evaluate_row(row):
            # Compute all metrics
            bleu = self.bleu_evaluation(row["target"], row["output"])
            bleu_nosan = self.bleu_evaluation(remove_san_from_text(row["target"]), remove_san_from_text(row["output"]))
            rouge1, rouge2, rougeL = self.rouge_evaluation(row["target"], row["output"])
            rouge1_nosan, rouge2_nosan, rougeL_nosan = self.rouge_evaluation(remove_san_from_text(row["target"]), remove_san_from_text(row["output"]))
            meteor = self.meteor_evaluation(row["target"], row["output"])
            meteor_nosan = self.meteor_evaluation(remove_san_from_text(row["target"]), remove_san_from_text(row["output"]))
            bertscore = self.bertscore_evaluation(row["target"], row["output"])
            bertscore_nosan = self.bertscore_evaluation(remove_san_from_text(row["target"]), remove_san_from_text(row["output"]))

            # Return the new scores as a dict
            return {
                "BLEU score": bleu/100,
                "ROUGE-1 score": rouge1,
                "ROUGE-2 score": rouge2,
                "ROUGE-L score": rougeL,
                "METEOR score": meteor,
                "BERTScore": bertscore,
                "BLEU score no SAN": bleu_nosan/100,
                "ROUGE-1 score no SAN": rouge1_nosan,
                "ROUGE-2 score no SAN": rouge2_nosan,
                "ROUGE-L score no SAN": rougeL_nosan,
                "METEOR score no SAN": meteor_nosan,
                "BERTScore no SAN": bertscore_nosan,
            }
        scores = [
            "BLEU score", "ROUGE-1 score", "ROUGE-2 score",
            "ROUGE-L score", "METEOR score", "BERTScore",
            "BLEU score no SAN", "ROUGE-1 score no SAN", "ROUGE-2 score no SAN",
            "ROUGE-L score no SAN", "METEOR score no SAN", "BERTScore no SAN"
        ]
        # Apply row-wise and overwrite columns directly
        df_results[scores] = df_results.progress_apply(__evaluate_row, axis=1, result_type="expand")

        if save_eval or show_graphs:
            def __create_hist_plot(column_name: str, min_val: float | None, max_val: float | None):
                fig, ax = plt.subplots()
                x = df_results[column_name].tolist()
                ax.hist(x)
                ax.set_xlim(min_val, max_val)
                ax.set_title(f"{column_name} distribution, avergage value: {np.average(x)}")
                return fig

            if save_eval:
                os.mkdir(folder_name)
                df_results.to_csv(os.path.join(folder_name, "result.csv"), index=False)
                with open(os.path.join(folder_name, 'scores.json'), 'w+') as outfile:
                    json.dump({f"avg {s}": np.average(df_results[s].tolist()) for s in scores}, outfile)

            for s in scores:
                fig = __create_hist_plot(s, 0.0, 1.0)
                if save_eval:
                    fig.savefig(os.path.join(folder_name, f"{safe_folder_name(s).lower()}.png"))

            if show_graphs:
                plt.show()

        return df_results
