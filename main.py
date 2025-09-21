import os
import argparse
import pandas as pd
from huggingface_hub import repo_exists

# These 2 lines prevent pytorch from trying to use Triton
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

from controller import Controller
from modules.utils import Debug, LLM
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    #############################################
    #               Parameters                  #
    #############################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=True, help='Show debug informations.')
    parser.add_argument('--llm', type=str, default="google/gemma-3-1b-it", help='LLM used as base model.')
    parser.add_argument('--dataset', type=str, default=None, help='Path to the dataset used for training/evaluation. If None, will create a dataset from content in ./data/raw/.')
    parser.add_argument('--trained_model', type=str, default=None, help='Path to a trained model. If None, will train a model with parameters given.')
    parser.add_argument('--llm_filter', type=str, default="google/gemma-3-1b-it", help='LLM used to filter/reformulate comments.')
    parser.add_argument('--evaluate_model', type=bool, default=True, help='evaluate the trained model with 20% of the dataset.')
    parser.add_argument('--evaluate_base_model', type=bool, default=True, help='evaluate the base model with 20% of the dataset.')
    parser.add_argument('--save_evaluation', type=bool, default=True, help='Will save the graphs and answers of the evaluation in a folder.')
    parser.add_argument('--show_evaluation', type=bool, default=False, help='Will show the graphs of the evaluation.')
    parser.add_argument('--prompt', type=str, default=None, help='Will answer to the prompt using the trained model. If no trained model given, then it will use the default llm.')
    # TODO: implement those functionalities
    # parser.add_argument('--generate_comments', type=str, default=None, help='Path to a pgn file.')
    # parser.add_argument('--pgn_analyzed', type=bool, default=False, help='Indicate that the PGN file given is already analyzed.')
    args = parser.parse_args()
    dbg = Debug(debug=args.debug)
    dbg.print("Debug: On")

    ctrl = Controller()


    #############################################
    #          Dataset configuration            #
    #############################################
    if args.dataset:
        dbg.print(f"USING {args.dataset} AS DATASET.")
        assert os.path.exists(args.dataset), f"dataset '{args.dataset}' doesn't exists !"
        assert os.path.isfile(args.dataset) and args.dataset.lower().endswith(".csv") , f"dataset '{args.dataset}' is not a csv file !"
        dataset_path = args.dataset
    else:
        dbg.print("CREATING DATASET FROM RAW DATA...\n")
        assert repo_exists(args.llm_filter), f"The LLM {args.llm_filter} doesn't exists/is not available !"
        dbg.print("\nSaving comments from games...\n")
        LLM(args.llm_filter)
        ctrl.save_good_comments_from_games()
        dbg.print("\nReformulating comments...\n")
        ctrl.reformulate_good_comments()
        dbg.print("\nSaving comments as a single csv file...\n")
        dataset_path = ctrl.save_comments_as_csv()

    #############################################
    #            Model configuration            #
    #############################################
    input_columns = ["moves", "engine_eval", "engine_best_line", "engine_best_alternative"]
    input_target = "reformulated"
    if args.trained_model:
        dbg.print("LOADING TRAINED MODEL.")
        assert os.path.exists(args.trained_model), f"Folder {args.trained_model} doesn't exists !"
        checkpoint_name = args.trained_model
    else:
        dbg.print("LOADING BASE LLM TO BE TRAINED.")
        assert repo_exists(args.llm), f"The LLM {args.llm} doesn't exists/is not available !"
        llm = LLM(args.llm)

        dbg.print("TRAINING LLM ON DATASET")
        dataset = pd.read_csv(dataset_path)
        if args.evaluate_model:
            dataset_train, dataset_eval = train_test_split(dataset, test_size=0.2, shuffle=True)
        else:
            dataset_train = dataset
        checkpoint_name = ctrl.train_model(
            dataset_train,
            input_columns,
            input_target
        )
    model, tokenizer = ctrl.load_model_and_tokenizer_from_checkpoint(checkpoint_name)

    #############################################
    #        Answering Prompt (optional)        #
    #############################################
    if args.prompt:
        dbg.print("ANSWERING PROMPT.")
        answer = ctrl.prompt_model(model, tokenizer, args.prompt)
        print("Prompt answer:")
        print(answer)


    #############################################
    #               Evaluation                  #
    #############################################
    if args.evaluate_model:
        dbg.print("EVALUATING TRAINED MODEL.")
        if args.trained_model is None:
            dataset = pd.read_csv(dataset_path)
            dataset_train, dataset_eval = train_test_split(dataset, test_size=0.2, shuffle=True)
        ctrl.evaluate_model(model, tokenizer, dataset_eval, input_columns, input_target, 4, args.save_evaluation, args.show_evaluation)

    if args.evaluate_base_model:
        dbg.print("EVALUATING BASE MODEL.")
        if not args.evaluate_model:
            dataset = pd.read_csv(dataset_path)
            dataset_train, dataset_eval = train_test_split(dataset, test_size=0.2, shuffle=True)
        assert repo_exists(args.llm), f"The LLM {args.llm} doesn't exists/is not available !"
        llm = LLM(args.llm)
        ctrl.evaluate_model(llm.model, llm.tokenizer, dataset_eval, input_columns, input_target, 4, args.save_evaluation, args.show_evaluation)
