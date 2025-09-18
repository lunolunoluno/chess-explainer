import json
import logging
import os.path
import re
import sys

from datetime import datetime
from typing import List, Any

import chess.pgn
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sympy.printing.pytorch import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling, PreTrainedTokenizerFast, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig

from modules.datautils import has_game_comments, pgn_to_id, get_all_pgn_files, get_all_comments_and_lines_in_game, \
    filter_good_comments, create_dataset, safe_folder_name, remove_return_char, create_generation_prompt
from modules.evaluator import Evaluator
from modules.utils import Debug, LLM
from modules.gameanalyzer import GameAnalyzer


class Controller:

    def __init__(self):
        self.dbg = Debug()

        self.data_path = os.getenv("DATA_PATH")
        assert os.path.exists(self.data_path), f"{self.data_path} doesn't exists !"
        self.data_raw_path = os.getenv("DATA_RAW_PATH")
        assert os.path.exists(self.data_raw_path), f"{self.data_raw_path} doesn't exists !"
        self.data_analyzed_path = os.getenv("DATA_ANALYZED_PATH")
        assert os.path.exists(self.data_analyzed_path), f"{self.data_analyzed_path} doesn't exists !"
        self.data_commented_path = os.getenv("DATA_COMMENTS_PATH")
        assert os.path.exists(self.data_commented_path), f"{self.data_commented_path} doesn't exists !"
        self.data_evaluations_path = os.getenv("DATA_EVALUATIONS_PATH")
        assert os.path.exists(self.data_evaluations_path), f"{self.data_evaluations_path} doesn't exists !"
        self.model_path = os.getenv("MODEL_PATH")
        assert os.path.exists(self.model_path), f"{self.model_path} doesn't exists !"

        # remove the non-critical logs in the terminal
        logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)

    def analyze_annotated_games(self) -> None:
        """
        Will take all the pgn files in DATA_RAW_PATH and analyzed all the games.
        Once analyzed, each game is saved in a pgn file in DATA_ANALYZED_PATH
        """
        ga = GameAnalyzer()

        pgn_files = get_all_pgn_files()
        self.dbg.print(f"Number of .pgn files to analyze in {self.data_raw_path}: {len(pgn_files)}")

        for pgnfile in tqdm(pgn_files):
            with open(pgnfile, "r", encoding="utf-8", errors="replace") as pgn_file:
                self.dbg.print(f"\nAnalyzing {pgnfile}...")
                while True:
                    try:
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            # No more game in the file so exit the while loop
                            break
                        if len(game.errors) > 0:
                            raise Exception(f"error in {game.headers}\n" + "\n".join([str(e) for e in game.errors]))
                    except Exception as e:
                        self.dbg.print(f"\nError parsing a game in {pgnfile}: {e}", file=sys.stderr)
                        continue  # skip to the next game in pgnfile

                    str_exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=True)
                    pgn_str = game.accept(str_exporter)
                    if has_game_comments(pgn_str):
                        self.dbg.print(f"Analyzing {game.headers}...")
                        game_id = pgn_to_id(pgn_str)
                        analyzed_game_path = os.path.join(self.data_analyzed_path, f"{game_id}.pgn")
                        if not os.path.exists(analyzed_game_path):
                            ga.analyze_game(game)
                            with open(analyzed_game_path, "w", encoding="utf-8") as new_pgn:
                                exporter = chess.pgn.FileExporter(new_pgn)
                                game.accept(exporter)
                            self.dbg.print(f"\tGame saved as {analyzed_game_path}")
                        else:
                            self.dbg.print(f"\tGame already saved as {analyzed_game_path}")
                    else:
                        self.dbg.print(f"No comments in {pgnfile} -> {game.headers}")

    def save_good_comments_from_games(self) -> None:
        """
        Will take all the pgn files in DATA_RAW_PATH and extract the comments that explain any player's mistake.
        All the comments from a game will be saved in a csv file in DATA_COMMENTS_PATH
        """
        llm = LLM()
        pipe = llm.get_pipe()

        # get all the pgn files that will be evaluated
        pgn_files = get_all_pgn_files()
        self.dbg.print(f"Number of .pgn files to analyze in {self.data_raw_path}: {len(pgn_files)}")

        for pgnfile in tqdm(pgn_files):
            self.dbg.print(f"Analyzing {pgnfile}...")

            # Read game in pgn file
            with open(pgnfile, encoding="utf-8", errors="replace") as pgn_game:
                game = chess.pgn.read_game(pgn_game)
                while game is not None:
                    header = game.headers
                    if 'White' in header and header['White'].strip() != '':
                        context = f"This is a game between {header['White']} (as White)"
                    else:
                        context = "This is a game between an unknown player (as White)"
                    if 'Black' in header and header['Black'].strip() != '':
                        context += f" and {header['Black']} (as Black)"
                    else:
                        context += " and an unknown player (as Black)"
                    str_exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=True)
                    pgn_str = game.accept(str_exporter)
                    csv_path = os.path.join(self.data_commented_path, f"{pgn_to_id(pgn_str)}.csv")
                    if not os.path.exists(csv_path):
                        comments = get_all_comments_and_lines_in_game(game, context)
                        if len(comments) > 0:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.reset_peak_memory_stats()
                            self.dbg.print(f"Analyzing {len(comments)} from {header}...")
                            good_comments = filter_good_comments(pipe, comments)
                            df = pd.DataFrame(good_comments)
                            df.to_csv(csv_path, index=False)

                    game = chess.pgn.read_game(pgn_game)

    def reformulate_good_comments(self) -> None:
        llm = LLM()
        pipe = llm.get_pipe()

        games = [
            os.path.join(self.data_commented_path, file)
            for file in os.listdir(self.data_commented_path)
            if os.path.isfile(os.path.join(self.data_commented_path, file)) and file.lower().endswith(".csv")
        ]

        for game_index, game_comments in enumerate(games):
            self.dbg.print(f"Reformulating comments in {game_comments}...")
            try:
                comments = pd.read_csv(game_comments)
            except pd.errors.EmptyDataError:
                self.dbg.print("Empty file !")
                continue
            except Exception as e:
                raise e

            if 'reformulated' in comments:
                # This means that this file has already been reformulated
                self.dbg.print("Already reformulated !")
                continue

            comments["reformulated"] = "-"
            good_comments = comments[comments["good"]]
            self.dbg.print(f"Reformulating {good_comments.shape[0]} comments...")

            if not good_comments.empty:
                prompt_model = lambda row: [
                    {"role": "system",
                     "content": """
                    Your job is to reformulate a comment made about a chess move.
                    The comment should be explaining why the move made is a mistake.
                    
                    Reformulate that comment to only keep the part that explains the mistake.
                    You may use some of the engine's information when reformulating but try to keep it as close to the original comment as possible.
                    While reformulating do the following too:
                    - The reformulated comment should only contain an explanation of the mistake.
                    - If the comment doens't suggest alternative lines, use the one provided by the engine.
                    - When using a pronoun to refer to a player, only use they/them/their.
                    - NEVER mention a player's name. Use either 'black' or 'white' according to the player's color.
                    - If the original comment is talking about something or someone unrelated to the game, do not mention it.
                    - If the comment isn't in english, translate the reformulation to english.
                    """},
                    {"role": "user",
                     "content": f"""
                    Context: {row['context']}
                    Engine evaluation: {row['engine_eval']}
                    Engine best line: {row['engine_best_line']}
                    Engine best alternative line: {row['engine_best_alternative']}
                    Here's the reason why this comment was picked: {row['reasoning']}
                    
                    Here's the comment to reformulate:
                    {row['comment']}
                    
                    Keep in mind while reformulating that:
                    - The reformulated comment should only contain an explanation of the mistake.
                    - If the comment doens't suggest alternative lines, use the one provided by the engine.
                    - When using a pronoun to refer to a player, only use they/them/their.
                    - NEVER mention a player's name. Use either 'black' or 'white' according to the player's color.
                    - If the original comment is talking about something or someone unrelated to the game, do not mention it.
                    - If the comment isn't in english, translate the reformulation to english.
                    
                    Only answer with the reformulated comment and nothing else.
                    """}
                ]

                prompts = [prompt_model(comment) for _, comment in
                           good_comments.iterrows()]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                outputs = pipe(prompts, batch_size=4)
                out_reformulated = [o[0]['generated_text'][-1]['content'] for o in outputs]
                good_comments.loc[:, "reformulated"] = out_reformulated
                comments.update(good_comments)

            comments.to_csv(games[game_index], index=False)

    def save_comments_as_csv(self) -> str:
        comments_files = [
            os.path.join(self.data_commented_path, file)
            for file in os.listdir(self.data_commented_path)
            if os.path.isfile(os.path.join(self.data_commented_path, file)) and file.lower().endswith(".csv")
        ]

        # Remove empty files
        for file_path in comments_files:
            try:
                pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                os.remove(file_path)
                self.dbg.print(f"Deleted empty CSV: {file_path}")

        # Get all the files without the deleted ones
        comments_files = [
            os.path.join(self.data_commented_path, file)
            for file in os.listdir(self.data_commented_path)
            if os.path.isfile(os.path.join(self.data_commented_path, file)) and file.lower().endswith(".csv")
        ]

        df_merged = pd.concat((pd.read_csv(f) for f in comments_files), ignore_index=True)
        merge_path = os.path.join(self.data_path, f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df_merged.to_csv(merge_path, index=False)

        df_filtered_merged = df_merged[df_merged["good"]]
        filtered_merged_path = os.path.join(self.data_path,
                                            f"filtered_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df_filtered_merged.to_csv(filtered_merged_path, index=False)

        self.dbg.print(f"{df_merged.shape[0]} total comments for {df_filtered_merged.shape[0]} good comments")

        return filtered_merged_path

    def train_model(self, train_dataset: pd.DataFrame, inputs_columns: List[str], label_column: str) -> str:
        assert set(inputs_columns).issubset(train_dataset.columns), f"One of the inputs ({inputs_columns}) is not found in the dataset ({train_dataset.columns})"
        assert label_column in train_dataset.columns, f"{label_column} is not found in the dataset !"

        llm = LLM()
        model_id = llm.model_id

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Quantization setup for QLoRA
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,  # apply the quantization
            device_map="auto"
        )

        dataset = create_dataset(train_dataset, inputs_columns, label_column)

        def tokenize_function(examples):
            # Tokenize separately
            prompt_tokens = tokenizer(examples["prompt"], truncation=True, max_length=512)
            full_tokens = tokenizer(examples["full_text"], truncation=True, padding="max_length", max_length=512)

            # Copy labels
            labels = full_tokens["input_ids"].copy()

            # Mask out prompt tokens
            prompt_len = len(prompt_tokens["input_ids"])
            labels[:prompt_len] = [-100] * prompt_len

            full_tokens["labels"] = labels
            return full_tokens

        self.dbg.print("Tokenizing dataset...")
        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # LoRA configuration
        lora_config = LoraConfig(
            r=64,  # QLoRA papers often use higher rank (32–64) compare to the classic LoRA rank 8
            lora_alpha=16,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(self.model_path, "lora-multiinput-output"),
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=1,  # Small batch size for memory efficiency
            gradient_accumulation_steps=4,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            learning_rate=2e-4,  # lr for QLoRA -> 2e-4, LoRA -> 5e-4
            bf16=torch.cuda.is_bf16_supported(),  # use bf16 if supported, else fp16
            fp16=not torch.cuda.is_bf16_supported(),
            push_to_hub=False,
            report_to=None,  # Disable wandb/tensorboard logging
        )

        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            processing_class=tokenizer,
            data_collator=data_collator
        )

        self.dbg.print("Starting training...")
        trainer.train()

        checkpoint_name = os.path.join(self.model_path, f"trained-{safe_folder_name(model_id)}-{datetime.today().strftime('%Y%m%d%H%M%S')}")
        model.save_pretrained(checkpoint_name)
        tokenizer.save_pretrained(checkpoint_name)

        self.dbg.print(f"Training done and saved as {checkpoint_name}!")
        return checkpoint_name

    def load_model_and_tokenizer_from_checkpoint(self, checkpoint_name: str) -> (PeftModel, PreTrainedTokenizerFast):
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
        peft_config = PeftConfig.from_pretrained(checkpoint_name)
        base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, checkpoint_name)

        return model, tokenizer

    def prompt_model(self, model: Any, tokenizer: Any, prompt: str) -> str:
        model.eval()

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                min_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def evaluate_model(self, model: Any, tokenizer: Any, test_dataset: pd.DataFrame, input_columns: list[str], input_target: str, batch_size: int = 1, save_eval: bool = False, show_graphs: bool = False) -> list:
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        prompts = []
        for _, row in test_dataset.iterrows():
            # info = "\n".join([f"{col}: {row[col]}" for col in input_columns])
            #
            # p = f"""
            #     Based on the following information:
            #     {info}
            #     Here is a concise explanation on why the last move played was a mistake:
            # """.strip()
            # p = re.sub(r'\t| {2,}', '', p)
            p = create_generation_prompt(row, input_columns)
            prompts.append({
                "prompt": p,
                "target": row[input_target]
            })

        results = []
        self.dbg.print("MODEL ANSWERING PROMPTS")
        for i in tqdm(range(0, len(prompts), batch_size)):
            end_range = i + batch_size if i + batch_size < len(prompts) else len(prompts)
            batch = [prompts[i]["prompt"] for i in range(i, end_range)]
            batch_targets = [prompts[i]["target"] for i in range(i, end_range)]

            token_lengths = [len(tokenizer.encode(b)) for b in batch]
            max_token = max(token_lengths)
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_token
            ).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    min_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id
                )

            decoded_outputs = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
            res = [{
                "prompt": p,
                "output": remove_return_char(o.removeprefix(p).strip()),
                "target": remove_return_char(t)
            }
                for p, t, o in zip(batch, batch_targets, decoded_outputs)
            ]
            results.extend(res)

        ev = Evaluator()

        self.dbg.print("EVALUATING MODEL ANSWERS")
        for res in tqdm(results):
            bleu = ev.bleu_evaluation(res["target"], res["output"])
            rouge1, rouge2, rougeL = ev.rouge_evaluation(res["target"], res["output"])
            meteor = ev.meteor_evaluation(res["target"], res["output"])
            bertscore = ev.bertscore_evaluation(res["target"], res["output"])
            res["BLEU score"] = bleu
            res["ROUGE-1 score"] = rouge1
            res["ROUGE-2 score"] = rouge2
            res["ROUGE-L score"] = rougeL
            res["METEOR score"] = meteor
            res["BERTScore"] = bertscore

        df_results = pd.DataFrame(results)
        if save_eval or show_graphs:
            def __create_hist_plot(column_name: str, min_val: float | None, max_val: float | None):
                fig, ax = plt.subplots()
                x = df_results[column_name].tolist()
                ax.hist(x)
                ax.set_xlim(min_val, max_val)
                ax.set_title(f"{column_name} distribution, avergage value: {np.average(x)}")
                return fig

            fig_bleu = __create_hist_plot("BLEU score", 0.0, 100.0)
            fig_rouge1 = __create_hist_plot("ROUGE-1 score", 0.0, 1.0)
            fig_rouge2 = __create_hist_plot("ROUGE-2 score", 0.0, 1.0)
            fig_rougeL = __create_hist_plot("ROUGE-L score", 0.0, 1.0)
            fig_meteor = __create_hist_plot("METEOR score", 0.0, 1.0)
            fig_bertscore = __create_hist_plot("BERTScore", None, None)

            if save_eval:
                folder_name = os.path.join(self.data_evaluations_path,f"{safe_folder_name(model.name_or_path)}_{datetime.today().strftime('%Y%m%d%H%M%S')}")
                os.mkdir(folder_name)
                df_results.to_csv(os.path.join(folder_name, "result.csv"), index=False)

                fig_bleu.savefig(os.path.join(folder_name, "bleu_score.png"))
                fig_rouge1.savefig(os.path.join(folder_name, "rouge1_score.png"))
                fig_rouge2.savefig(os.path.join(folder_name, "rouge2_score.png"))
                fig_rougeL.savefig(os.path.join(folder_name, "rougel_score.png"))
                fig_meteor.savefig(os.path.join(folder_name, "meteor_score.png"))
                fig_bertscore.savefig(os.path.join(folder_name, "bertscore.png"))

                with open(os.path.join(folder_name, 'scores.json'), 'w+') as outfile:
                    json.dump({
                        "avg BLEU score": np.average(df_results["BLEU score"].tolist()),
                        "avg ROUGE-1 score": np.average(df_results["ROUGE-1 score"].tolist()),
                        "avg ROUGE-2 score": np.average(df_results["ROUGE-2 score"].tolist()),
                        "avg ROUGE-L score": np.average(df_results["ROUGE-L score"].tolist()),
                        "avg METEOR score": np.average(df_results["METEOR score"].tolist()),
                        "avg BERTScore": np.average(df_results["BERTScore"].tolist())
                    }, outfile)

            if show_graphs:
                plt.show()

        return results
