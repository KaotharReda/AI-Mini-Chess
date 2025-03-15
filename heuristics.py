
# File for all the heuristic functions used in the game

"""
Evaluates the game state using heuristic function and returns the score (e0 implemented currently)    

Args: 
    - game_state:   dictionary | Dictionary representing the current game state
Returns:
    - integer value representing the score of the game state
"""
def heuristic_e0(self, game_state):
    piece_values = {'p': 1, 'N': 3, 'B': 3, 'Q': 9, 'K': 999} # Piece values for evaluation function e0
    white_score, black_score = 0, 0
    for row in game_state["board"]:
        for piece in row:
            if piece == '.': # Zero value
                continue
            if piece[0] == 'w':
                white_score += piece_values[piece[1]] # Strip color, add value from piece_values dict
            else:
                black_score += piece_values[piece[1]]
    return white_score - black_score

# Values based on https://www.chessprogramming.org/Simplified_Evaluation_Function
# This also gave me inspiration for the piece-square tables

PIECE_VALUES = {
    'p': 100,
    'N': 320,
    'B': 330,
    'R': 500,
    'Q': 900,
    'K': 20000
}

PIECE_SQUARE_TABLES_WHITE = {
    'p': [  
        [60,  60,  60,  60, 60],
        [50,  50,  50,  50,  50],
        [0,  30,  40,  30,  0],
        [0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0]  
    ],
    'N': [  
        [-50, -30, -30, -30, -50],
        [-30,  20,  30,  20, -30],
        [-30,  30,  50,  30, -30],
        [-30,  20,  30,  20, -30],
        [-50, -30, -30, -30, -50]
    ],
    'B': [
        [-20, -10,  0, -10, -20],
        [-10,  30,  30,  30, -10],
        [0,   30,  50,  30,  0],
        [-10,  30,  30,  30, -10],
        [-20, -10,  0, -10, -20]
    ],
    'Q': [  
        [-20,  -10,  -5,  -10, -20],
        [-10,   40,  40,  40,  -10],
        [-5,   50,  50,  50,  -5],
        [-10,   40,  40,  40,  -10],
        [-20,  -10,  -5,  -10, -10]
    ],
    'K': [  
        [-30, -20,  -20, -20, -30],
        [-30, -20,  -20, -20, -30],
        [-30, -20,  -20, -20, -30],
        [15,  10,  15,  10, 15],
        [20, 30, 10, 30, 20]
    ]
}

PIECE_SQUARE_TABLES_BLACK = {
    'p': [  
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 30, 40, 30, 0],
        [50, 50, 50, 50, 50],
        [60, 60, 60, 60, 60]
    ],
    'N': [  
        [-50, -30, -30, -30, -50],
        [-30, 20, 30, 20, -30],
        [-30, 30, 50, 30, -30],
        [-30, 20, 30, 20, -30],
        [-50, -30, -30, -30, -50]
    ],
    'B': [
        [-20, -10, 0, -10, -20],
        [-10, 30, 30, 30, -10],
        [0, 30, 50, 30, 0],
        [-10, 30, 30, 30, -10],
        [-20, -10, 0, -10, -20]
    ],
    'Q': [  
        [-20, -10, -5, -10, -20],
        [-10, 40, 40, 40, -10],
        [-5, 50, 50, 50, -5],
        [-10, 40, 40, 40, -10],
        [-20, -10, -5, -10, -20]
    ],
    'K': [  
        [20, 30, 10, 30, 20],
        [15, 10, 15, 10, 15],
        [-30, -20, -20, -20, -30],
        [-30, -20, -20, -20, -30],
        [-30, -20, -20, -20, -30]
    ]
}

"""
    Heuristic function that evaluates the game state based on the material value of the pieces
    and the piece-square tables (position). The heuristic value is calculated as the sum of the material
    value of the pieces and the value of the piece-square tables for the pieces on the board.

    Args:
        game_state: The game state to evaluate.

    Returns:
        The heuristic value of the game state.
    """
def heuristic_e1(game_state):
    board = game_state["board"]
    heuristic_value = 0
    
    for row in range(5):
        for col in range(5):
            piece = board[row][col]
            if piece != '.':
                color = piece[0]
                piece = piece[1]
                if color == 'w':
                    piece_value = PIECE_VALUES[piece]
                    piece_square_value = PIECE_SQUARE_TABLES_WHITE[piece][row][col]
                    heuristic_value += piece_value + piece_square_value
                else:
                    piece_value = PIECE_VALUES[piece]
                    piece_square_value = PIECE_SQUARE_TABLES_BLACK[piece][row][col]
                    heuristic_value -= (piece_value + piece_square_value) # Subtract from the total, black aims to minimize

    return heuristic_value

test_state = {
            "board": 
            [['bK', '.', '.', 'bN', '.'],
            ['.', '.', 'wp', '.', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', '.', 'wB', '.', '.']],
            "turn": 'white',
            "move_count": 0
    }