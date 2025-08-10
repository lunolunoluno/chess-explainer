import os
import re
from controller import Controller
from modules.utils import Debug, LLM

if __name__ == "__main__":
    dbg = Debug(debug=True)
    dbg.print("Debug: On")
    llm = LLM()

    ctrl = Controller()
    dbg.print("\nSAVE GOOD COMMENTS FROM GAMES\n"+"="*40)
    ctrl.save_good_comments_from_games()
    dbg.print("\nREFORMULATE COMMENTS\n"+"="*40)
    ctrl.reformulate_good_comments()
    dbg.print("\nSAVE COMMENTS AS A SINGLE CSV\n"+"="*40)
    dataset_path = ctrl.save_comments_as_csv()

    checkpoint_name = ctrl.train_model(dataset_path, ["context", "moves", "engine_eval", "engine_best_line"], "reformulated")
    model, tokenizer = ctrl.load_model_and_tokenizer_from_checkpoint(checkpoint_name)

    prompt = re.sub(r'\t| {2,}', '', f"""
                        Based on the following information:
                        context: This is a game between Scicluna, Tristan Jes (as White) and Noel, Lucien (as Black). Last move played: White plays pawn to d3
                        moves: 1. e4 e5 2. Nf3 Nf6 3. Nxe5 Nc6 4. Nxc6 dxc6 5. Nc3 Bc5 6. d3
                        engine_eval: [%eval -0.93]
                        engine_best_line: 6. h3 O-O 7. d3 b5 8. Be2
                        Generate a comment explaining the error that the player just made
                        Comment: \n""".strip())

    comment = ctrl.prompt_model(model, tokenizer, prompt)

    # The comment should say something like White should have played 6. h3 preventing Black from playing Ng4
    print(comment)


