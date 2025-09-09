import os

import pandas as pd

# These 2 lines prevent pytorch from trying to use Triton
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

from controller import Controller
from modules.utils import Debug, LLM
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    dbg = Debug(debug=True)
    dbg.print("Debug: On")

    llm = LLM("Qwen/Qwen3-0.6B")

    ctrl = Controller()
    dataset_path = os.path.join(".", "Notebooks", "reformulated_data_20250902_163137.csv")
    dataset = pd.read_csv(dataset_path)
    dataset_train, dataset_eval = train_test_split(dataset, test_size=0.2, shuffle=True)

    input_columns = ["moves", "engine_eval", "engine_best_line", "engine_best_alternative"]
    input_target = "reformulated"

    checkpoint_name = ctrl.train_model(
        dataset_train,
        input_columns,
        input_target
    )
    # checkpoint_name = os.path.join(".", "models", "trained-google_gemma-3-1b-it-20250907213056")
    model, tokenizer = ctrl.load_model_and_tokenizer_from_checkpoint(checkpoint_name)

    # evaluate the model after training
    ctrl.evaluate_model(model, tokenizer, dataset_eval, input_columns, input_target, 2, True, False)

    # evaluate the base model with no training
    ctrl.evaluate_model(llm.model, llm.tokenizer, dataset_eval, input_columns, input_target, 2, True, False)

