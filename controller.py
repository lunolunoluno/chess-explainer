import re
import os.path
import sys
import gc
import torch
from typing import List
from datetime import datetime

import chess.pgn
import logging
import pandas as pd

from sympy.printing.pytorch import torch
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling, AutoModelForSeq2SeqLM, PreTrainedTokenizerFast
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig

from modules.datautils import has_game_comments, pgn_to_id, get_all_pgn_files, get_all_comments_and_lines_in_game, \
    filter_good_comments, create_dataset
from modules.utils import Debug, LLM
from modules.gameanalyzer import GameAnalyzer






class Controller:

    def __init__(self):
        self.dbg = Debug()

        self.data_raw_path = os.getenv("DATA_RAW_PATH")
        assert os.path.exists(self.data_raw_path), f"{self.data_raw_path} doesn't exists !"
        self.data_analyzed_path = os.getenv("DATA_ANALYZED_PATH")
        assert os.path.exists(self.data_analyzed_path), f"{self.data_analyzed_path} doesn't exists !"
        self.data_commented_path = os.getenv("DATA_COMMENTS_PATH")
        assert os.path.exists(self.data_commented_path), f"{self.data_commented_path} doesn't exists !"

        # remove the non-critical logs in the terminal
        logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)

    def analyze_annotated_games(self) -> None:
        """
        Will take all the pgn files in DATA_RAW_PATH and analyzed all the games.
        Once analyzed, each game is saved in a pgn file in DATA_ANALYZED_PATH
        """
        ga = GameAnalyzer()

        pgn_files = get_all_pgn_files()
        print(f"Number of .pgn files to analyze in {self.data_raw_path}: {len(pgn_files)}")

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
                        print(f"\nError parsing a game in {pgnfile}: {e}", file=sys.stderr)
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
        Once analyzed, each game is saved in a pgn file in DATA_ANALYZED_PATH
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
            pgn_game = open(pgnfile)
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
                cvs_path = os.path.join(self.data_commented_path, f"{pgn_to_id(pgn_str)}.csv")
                if not os.path.exists(cvs_path):
                    comments = get_all_comments_and_lines_in_game(game, context)
                    if len(comments) > 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                        self.dbg.print(f"Analyzing {len(comments)} from {header}...")
                        good_comments = filter_good_comments(pipe, comments)
                        df = pd.DataFrame(good_comments)
                        df.to_csv(cvs_path, index=False)

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
            comments = pd.read_csv(game_comments)
            self.dbg.print(f"Reformulating comments in {game_comments}...")
            if 'reformulated' in comments:
                # This means that this file has already been reformulated
                self.dbg.print("Already reformulated !")
                continue

            comments["reformulated"] = "-"
            good_comments = comments[comments["good"]]
            self.dbg.print(f"Reformulating {good_comments.shape[0]} comments...")

            if not good_comments.empty:
                prompt_model = lambda context, comment: [
                    {"role": "system",
                     "content": """You're job is to reformulate chess annotations to make them cleaner. 
                    Additionally, it is VERY IMPORTANT that when reformulating you do the following:  
                    -   When using a pronoun to refer to a player, only use they/them/their
                    -   When mentionning a player's name, use either 'black' or 'white' according to the player's color."""},
                    {"role": "user",
                     "content": f"""Here is a small context regarding the game:
                     '{context}'
                     You don't have to mention anything about the context in the reformulated comment unless deemed necessary.
                     Here is the comment that you have to reformulate:
                     '{comment}'
                     Remember that you have to simplify the comment to only keep the explaination as to why the move played was bad.
                     Additionnaly, remember that  it is VERY IMPORTANT that when reformulating you do the following: 
                    -   When using a pronoun to refer to a player, only use they/them/their
                    -   When mentionning a player's name, use either 'Black' or 'White' according to the player's color."""}
                ]

                prompts = [prompt_model(comment['context'], comment['comment']) for _, comment in
                           good_comments.iterrows()]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                outputs = pipe(prompts, batch_size=4)
                out_reformulated = [o[0]['generated_text'][-1]['content'] for o in outputs]
                good_comments.loc[:, "reformulated"] = out_reformulated
                comments.update(good_comments)

            comments.to_csv(games[game_index], index=False)


    def train_model(self, dataset_path: str, inputs_columns: List[str], label_column: str) -> str:
        assert os.path.exists(dataset_path), f"{dataset_path} does not exists !"
        df_dataset = pd.read_csv(dataset_path)
        assert set(inputs_columns).issubset(df_dataset.columns), f"One of the inputs ({inputs_columns}) is not found in the dataset ({df_dataset.columns})"
        assert label_column in df_dataset.columns, f"{label_column} is not found in the dataset !"

        llm = LLM()
        model_id = llm.model_id

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        dataset = create_dataset(dataset_path, inputs_columns, label_column)

        def tokenize_function(examples):
            return tokenizer(
                examples["full_text"],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )

        self.dbg.print("Tokenizing dataset...")
        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # LoRA configuration
        lora_config = LoraConfig(
            r=8,
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
            output_dir="./lora-multiinput-output",
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=1,  # Small batch size for memory efficiency
            gradient_accumulation_steps=4,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            learning_rate=5e-4,
            fp16=True,  # Use mixed precision training
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

        checkpoint_name = f"trained-{model_id.split('/')[-1:]}-{datetime.today().strftime('%Y%m%d%H%M%S')}"
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

    def prompt_model(self, model, tokenizer, prompt)->str:
        model.eval()

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)

        return tokenizer.decode(outputs[0], skip_special_tokens=True)
