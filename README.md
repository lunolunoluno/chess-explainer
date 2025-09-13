# THIS IS STILL A WORK IN PROGRESS

# Chess Explainer
## Project Structure
```plaintext
├── Notebooks/      # Juypter Notebooks with alternative codes to filter and reformulate data
├── data/           
    ├──comments/    # All comments extracted from games
    ├──evaluations/ # Results of model evaluation
    ├──raw/         # All raw .pgn files to create the dataset
    ├──*.py         # Utility scripts used to manipulate the dataset
    └──*.csv        # Partial or complete datasets
├── models/         # Weights of trained models
├── modules/        # Modules of the project
├── controller.py   # Contains the main functions of the project
└── main.py         # Entry point of the code
```

# Requirements
This code was made with **python version 3.10**. I haven't tested it with any other versions.

This code was made to run on a computer with **CUDA version >= 12.8**.
It might work with other versions of CUDA but I haven't tested it and it requires to install the right [version of PyTorch](https://pytorch.org/get-started/locally/).
The code should also work on a computer with no GPU although it is not recommended.

## Install requirements
The list of the required libraries is in the **requirements.txt** file and can be installed with : 
```shell
pip install -r ./requirements.txt
```

## Set up .env file
You can copy the `.env.example` file to create your `.env` and fill it with the proper values.
```js
ENGINE_PATH="/path/to/chess/engine"

DATA_PATH="./data"
DATA_RAW_PATH="./data/raw"
DATA_ANALYZED_PATH="./data/analyzed"
DATA_COMMENTS_PATH="./data/comments"
DATA_EVALUATIONS_PATH="./data/evaluations"
MODEL_PATH="./models"

HUGGING_FACE_TOKEN="TOKEN HERE"
```

### Chess engine
You can download Stockfish [here](https://stockfishchess.org/) and set `ENGINE_PATH` to the path were you install it. 



