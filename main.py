import os
import re
from controller import Controller
from modules.utils import Debug, LLM

if __name__ == "__main__":
    dbg = Debug(debug=True)
    dbg.print("Debug: On")
    llm = LLM()

    ctrl = Controller()
    # ctrl.save_good_comments_from_games()
    # ctrl.reformulate_good_comments()

    dataset_path = os.path.join(".", "data", "filtered_merged_20250708_134716.csv")

    # checkpoint_name = ctrl.train_model(dataset_path, ["context", "moves"], "reformulated")
    checkpoint_name = "trained-google/gemma-3-1b-it-20250809144458"
    model, tokenizer = ctrl.load_model_and_tokenizer_from_checkpoint(checkpoint_name)

    prompt = re.sub(r'\t| {2,}', '', f"""
                        Based on the following information:
                        context: This is a game between Scicluna, Tristan Jes (as White) and Noel, Lucien (as Black). Last move played: White plays pawn to d3
                        moves: 1. e4 e5 2. Nf3 Nf6 3. Nxe5 Nc6 4. Nxc6 dxc6 5. Nc3 Bc5 6. d3
                        Generate a comment explaining the error that the player just made
                        Comment: \n""".strip())

    comment = ctrl.prompt_model(model, tokenizer, prompt)

    print(comment)


