import nltk
import sacrebleu

from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score


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
