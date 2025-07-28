# Requirements
This code was made with **python version 3.10**. I haven't tested it with any other versions.

This code was made to run on a computer with **CUDA version >= 12.8**.
It might work with other versions of CUDA but I haven't tested it and it requires to install the right [version of PyTorch](https://pytorch.org/get-started/locally/).

The list of the required libraries is in the **requirements.txt** file and can be installed with : 
```shell
pip install -r ./requirements.txt
```

# TO-DO LIST

## Create dataset

- 🔄 Finding games with comments on inaccuracies, mistakes and blunders
- 🔄 Create a process to filter/improve the comments on these games
  - ✅ Find all comments made on inaccuracies, mistakes and blunders
  - Add name of the opening in dataset
  - Add Stockfish evaluation in the dataset
  - (Maybe) Add Maia evaluation in the dataset
  - 🔄 Use LLM to evaluate if comment made is of good quality
  - 🔄 Use LLM to rephrase/improve comment

## Create AI agent

- Create agent using a pretrained LLM

## Create training process

- Train the agent to generate comments similar to the ones found on the dataset
- The comment will be made based on some data:
  - The moves leading up to the position
  - The best moves recommended by a strong engine ([Stockfish](https://stockfishchess.org/))
  - (Maybe) Moves recommended by a human-like AI ([Maia](https://www.maiachess.com/))

# Sources

- [Lichess.org source code to evaluate chess moves](https://github.com/lichess-org/lila/blob/cf9e10df24b767b3bc5ee3d88c45437ac722025d/modules/analyse/src/main/Advice.scala)
- [Stockfish Download](https://stockfishchess.org/download/)

## Annotated games

- [ValdemarOrn's Github](https://github.com/ValdemarOrn/Chess/blob/master/Annotated%20Games/)
- [Path to chess mastery's annotated games](https://www.pathtochessmastery.com/)
- [Lichess studies](https://lichess.org/study)