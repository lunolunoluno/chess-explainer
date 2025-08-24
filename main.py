import os

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# These 2 lines prevent pytorch from trying to use Triton
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

import re
from controller import Controller
from modules.utils import Debug, LLM
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    dbg = Debug(debug=True)
    dbg.print("Debug: On")

    ctrl = Controller()
    dataset_path = os.path.join(".", "data", "filtered_merged_20250708_134716.csv")
    dataset = pd.read_csv(dataset_path)
    dataset_train, dataset_eval = train_test_split(dataset, test_size=0.2, shuffle=True)

    # This will be used as temporary models but in the final product the trained model will be used
    model_id = "google/gemma-3-1b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    input_columns = ["moves", "context"] # TODO: replace with the correct list
    input_target = "reformulated"

    res = ctrl.evaluate_model(model, tokenizer, dataset_eval, input_columns, input_target, 2, True, True)
