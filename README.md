# AI-Mini-Chess

## Overview

Mini Chess is a simplified chess game implemented in Python. The game is played on a 5x5 board with a subset of standard chess pieces and rules, modified as follows:

- **King:** Moves one square in any direction. It can move into check, and capturing the opponent's king wins the game.
- **Queen:** Moves horizontally, vertically, or diagonally any number of squares (provided the path is clear).
- **Bishop:** Moves diagonally any number of squares (provided the path is clear).
- **Knight:** Moves in an L-shape (2 squares in one direction and 1 in the perpendicular direction) and can jump over pieces.
- **Pawn:** Moves one square forward and captures diagonally. When a pawn reaches the opposite end, it is promoted to a Queen.

Additional features include:
- **Warning System:** Players receive a warning for an invalid move. A second invalid move results in a loss.
- **Draw Conditions:** The game ends in a draw if the maximum number of turns is reached or if the current player has no valid moves (stalemate).

## How to Run

### Requirements
- Python 3.x

### Running the Game

1. **Use the following command to run the game from the command line**

   ```bash
python MiniChess.py --alpha-beta False --timeout 5 --max_turns 100 --play-mode H-H --heuristic e0

2. **Clone or Download the Repository**

   If you have Git installed, clone the repository:
   ```bash
   git clone https://github.com/KaotharReda/AI-Mini-Chess.git
   
