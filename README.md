# TO-DO LIST

## Create dataset

- Finding games with comments on inaccuracies, mistakes and blunders
- Create a process to filter/improve the comments on these games
  - Find all comments made on inaccuracies, mistakes and blunders
  - Use LLM to evaluate if comment made is of good quality
  - Use LLM to rephrase/improve comment

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
