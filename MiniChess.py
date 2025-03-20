import copy
import time
import argparse
import os
from heuristics import heuristic_e0, heuristic_e1, heuristic_e2

class MiniChess:
    def __init__(self, alpha_beta, timeout, max_turns, play_mode, heuristic):
        self.current_game_state = self.init_board()
        self.alpha_beta = alpha_beta
        self.timeout = timeout
        self.max_turns = max_turns
        self.play_mode = play_mode
        self.output_filename = f"gameTrace-{alpha_beta}-{timeout}-{max_turns}.txt"
        self.output_dir = "game_outputs"
        self.warnings = {"white": 0, "black": 0}
        self.cols = ['A', 'B', 'C', 'D', 'E'] # for displaying valid moves with correct notation
        self.players = {play_mode[0]: "white",  play_mode[2]: "black"}
        self.heuristic = heuristic
        self.states_explored = 0
        self.states_explored_by_depth = {}
        self.leaf_nodes = 0

    """
    Initialize the board

    Args:
        - None
    Returns:
        - state: A dictionary representing the state of the game
    """
    def init_board(self):
        state = {
                "board": 
                [['bK', 'bQ', 'bB', 'bN', '.'],
                ['.', '.', 'bp', 'bp', '.'],
                ['.', '.', '.', '.', '.'],
                ['.', 'wp', 'wp', '.', '.'],
                ['.', 'wN', 'wB', 'wQ', 'wK']],
                "turn": 'white',
                "move_count": 0
        }
        return state

    """
    Prints the board
    
    Args:
        - None
    Returns:
        - None
    """
    def display_board(self, game_state=None):
        if not game_state:
            game_state = self.current_game_state
        print()
        for i, row in enumerate(game_state["board"], start=1):
            print(str(6-i) + "  " + ' '.join(piece.rjust(3) for piece in row))
        print()
        print("     A   B   C   D   E")
        print()


    def board_to_string(self):
        board_str = ""
        for i, row in enumerate(self.current_game_state["board"], start=1):
            board_str += str(6 - i) + "  " + ' '.join(piece.rjust(3) for piece in row) + "\n"
        board_str += "\n     A   B   C   D   E\n\n"
        return board_str
    
    # Helper function to print cummulative states explored
    def _print_states_explored(self, f):
        states_explored_by_depth_str = "Cumulative States Explored by Depth: "
        states_explored_percent_str = "Cumulative States Explored by Depth (%): "
        for depth, count in self.states_explored_by_depth.items():
            if depth == 0:
                continue
            states_explored_by_depth_str += f"{depth}={count} "
            states_explored_percent_str += f"{depth}={(count*100/self.states_explored):.1f}% "
        f.write(states_explored_by_depth_str + "\n")
        f.write(states_explored_percent_str + "\n")

    # Helper function to print AI statistics
    def _print_ai_stats(self, f, search_score, time_for_action):
        f.write(f"Search Score: {search_score}\n")
        f.write(f"Board Heuristic Score: {self.heuristic(self.current_game_state)}\n")
        f.write("Time taken: {:0.2f} seconds\n".format(time_for_action))
        f.write(f"Cummulative States Explored: {self.states_explored}\n")
        self._print_states_explored(f)
        f.write(f"Branching Factor: {((self.states_explored-1)/(self.states_explored - self.leaf_nodes)):.1f}\n") 

    """
    Check if the move is valid    
    
    Args: 
        - game_state:   dictionary | Dictionary representing the current game state
        - move          tuple | the move which we check the validity of ((start_row, start_col),(end_row, end_col))
    Returns:
        - boolean representing the validity of the move
    """
    def is_valid_move(self, game_state, move):
        # Check if move is in list of valid moves

        valid_moves, _ = self.valid_moves(game_state) # Get all valid moves
        if move in valid_moves:
            return True
        return False

    """
    Returns a list of valid moves

    Args:
        - game_state:   dictionary | Dictionary representing the current game state
    Returns:
        - valid moves:   list | A list of nested tuples corresponding to valid moves [((start_row, start_col),(end_row, end_col)),((start_row, start_col),(end_row, end_col))]
    """
    def valid_moves(self, game_state):
        # Return a list of all the valid moves.
        # Implement basic move validation
        # Check for out-of-bounds, correct turn, move legality, etc
        valid_moves = []
        correct_notation_moves = [] # List of valid moves in correct notation for visual purposes
        for s_row in range(5):
            for s_col in range(5):
                piece = game_state["board"][s_row][s_col]
                if piece == '.' or piece[0] != game_state["turn"][0]:
                    continue

                start = f"{self.cols[s_col]}{5 - s_row}"  # Convert start to correct notation

                def add_move(e_row, e_col):
                    """ Helper function to add a move in the correct format """
                    end = f"{self.cols[e_col]}{5 - e_row}"
                    correct_notation_moves.append((start, end))

                # Generate valid moves for each piece
                if piece[1] == 'K':  # King
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            e_row, e_col = s_row + dr, s_col + dc
                            if 0 <= e_row < 5 and 0 <= e_col < 5:
                                target = game_state["board"][e_row][e_col]
                                if target == '.' or target[0] != piece[0]:
                                    valid_moves.append(((s_row, s_col), (e_row, e_col))) # format ((start_row, start_col),(end_row, end_col))
                                    add_move(e_row, e_col)

                elif piece[1] == 'Q':  # Queen
                    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
                    for dr, dc in directions:
                        e_row, e_col = s_row + dr, s_col + dc
                        while 0 <= e_row < 5 and 0 <= e_col < 5:
                            target = game_state["board"][e_row][e_col]
                            if target == '.' or target[0] != piece[0]:
                                valid_moves.append(((s_row, s_col), (e_row, e_col)))
                                add_move(e_row, e_col)
                            if target != '.':
                                break
                            e_row += dr
                            e_col += dc

                elif piece[1] == 'B':  # Bishop
                    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                    for dr, dc in directions:
                        e_row, e_col = s_row + dr, s_col + dc
                        while 0 <= e_row < 5 and 0 <= e_col < 5:
                            target = game_state["board"][e_row][e_col]
                            if target == '.' or target[0] != piece[0]:
                                valid_moves.append(((s_row, s_col), (e_row, e_col)))
                                add_move(e_row, e_col)
                            if target != '.':
                                break
                            e_row += dr
                            e_col += dc

                elif piece[1] == 'N':  # Knight
                    moves = [(-2, -1), (-1, -2), (-2, 1), (-1, 2), (2, -1), (1, -2), (2, 1), (1, 2)]
                    for dr, dc in moves:
                        e_row, e_col = s_row + dr, s_col + dc
                        if 0 <= e_row < 5 and 0 <= e_col < 5:
                            target = game_state["board"][e_row][e_col]
                            if target == '.' or target[0] != piece[0]:
                                valid_moves.append(((s_row, s_col), (e_row, e_col)))
                                add_move(e_row, e_col)

                elif piece[1] == 'p':  # Pawn
                    direction = -1 if piece[0] == 'w' else 1
                    # Move forward
                    e_row, e_col = s_row + direction, s_col
                    if 0 <= e_row < 5 and game_state["board"][e_row][e_col] == '.':
                        valid_moves.append(((s_row, s_col), (e_row, e_col)))
                        add_move(e_row, e_col)
                    # Capture diagonally
                    for dc in [-1, 1]:
                        e_row, e_col = s_row + direction, s_col + dc
                        if 0 <= e_row < 5 and 0 <= e_col < 5:
                            target = game_state["board"][e_row][e_col]
                            if target != '.' and target[0] != piece[0]:
                                valid_moves.append(((s_row, s_col), (e_row, e_col)))
                                add_move(e_row, e_col)

        return valid_moves, correct_notation_moves


    """
    Modify to board to make a move

    Args: 
        - game_state:   dictionary | Dictionary representing the current game state
        - move          tuple | the move to perform ((start_row, start_col),(end_row, end_col))
    Returns:
        - game_state:   dictionary | Dictionary representing the modified game state
    """
    def make_move(self, game_state, move):
        ((start_row, start_col), (end_row, end_col)) = move
        piece = game_state["board"][start_row][start_col]
        game_state["board"][start_row][start_col] = '.'

        # Promotion logic
        if piece[1] == 'p':
            # For white: promotion when reaching top (index 0)
            if piece[0] == 'w' and end_row == 0:
                piece = 'wQ'
            # For black: promotion when reaching bottom (index 4)
            elif piece[0] == 'b' and end_row == 4:
                piece = 'bQ'

        game_state["board"][end_row][end_col] = piece
        game_state["turn"] = "black" if game_state["turn"] == "white" else "white"
        game_state["move_count"] += 1
        return game_state

    def is_straight_move(self, s_row, s_col, e_row, e_col):
        return s_row == e_row or s_col == e_col

    def is_diagonal_move(self, s_row, s_col, e_row, e_col):
        return abs(s_row - e_row) == abs(s_col - e_col)

    """
    Parse the input string and modify it into board coordinates

    Args:
        - move: string representing a move "B2 B3"
    Returns:
        - (start, end)  tuple | the move to perform ((start_row, start_col),(end_row, end_col))
    """
    def parse_input(self, move):
        try:
            start, end = move.split()
            start = (5-int(start[1]), ord(start[0].upper()) - ord('A'))
            end = (5-int(end[1]), ord(end[0].upper()) - ord('A'))
            return (start, end)
        except:
            return None

    def convert_to_notation(self, pos):
        row, col = pos
        return f"{chr(col + ord('A'))}{5 - row}"
    
    def convert_2_notation(self, move):
       ((s_row, s_col), (e_row, e_col)) = move
       start = f"{self.cols[s_col]}{5 - s_row}"  # Convert start to correct notation
       end = f"{self.cols[e_col]}{5 - e_row}"
       return (start, end)

    def is_king_captured(self, game_state):
        current_turn = game_state["turn"]
        opponent = "black" if current_turn == "white" else "white"
        king = f"{opponent[0]}K"
        for row in game_state["board"]:
            if king in row:
                return False
        return True

    """
    Minimax algorithm to find the best move.
    
    Args:
        - game_state: dictionary | Current game state
        - depth: int | Depth to search in the game tree
        - maximizing_player: bool | True if maximizing, False if minimizing
        - alpha_beta: bool | Enable alpha-beta pruning
        - alpha: float | Alpha value for pruning (only if alpha-beta is enabled)
        - beta: float | Beta value for pruning (only if alpha-beta is enabled)
    
    Returns:
        - best_score: int | Evaluation score of the best move
        - best_move: tuple | The best move in ((start_row, start_col), (end_row, end_col)) format
    """
    def minimax(self, game_state, max_depth, current_depth, start_time, alpha_beta=False, alpha=float('-inf'), beta=float('inf')):
        
        # increment counters
        self.states_explored +=1
        if current_depth not in self.states_explored_by_depth:
            self.states_explored_by_depth[current_depth] = 1
        else:
            self.states_explored_by_depth[current_depth] += 1

        # Check if timeout has been reached
        if time.time() - start_time >= self.timeout:
            return self.heuristic(game_state), None, None
        
        # Check if we reached the maximum depth
        if current_depth == max_depth:
            self.leaf_nodes += 1 # increment leaf node counter to calculate branching factor
            return self.heuristic(game_state), None, None
        
        # Generate all valid moves for the current game state
        valid_moves, _ = self.valid_moves(game_state)

        # No more valid moves, we reached a stalemate
        if not valid_moves:
            return self.heuristic(game_state), None , None  

        best_move = None
        is_maximizing = game_state["turn"] == "white"
        best_score = float('-inf') if is_maximizing else float('inf') # set root to initial alpha (-inf) or beta (inf)
        # white's turn
        
        for move in valid_moves: # Evaluate all possible moves
            # create deep copy of state with next move
            new_state = self.make_move(copy.deepcopy(game_state), move)

            # Check timeout before recursing
            if time.time() - start_time >= self.timeout: # might need to include some time to break out (5ms??)
                break # stop searching since time is up

            # Recursively evaluate the new state
            score, _ , _ = self.minimax(new_state, max_depth, current_depth+1, start_time, alpha_beta, alpha, beta) 
            

            if is_maximizing:
                if score > best_score: # store max score
                    best_score = score
                    best_move = move
                if alpha_beta:
                    alpha = max(alpha, best_score)
                    if beta <= alpha: # we can prune branches from a min descendant that has a beta <= n(alpha)
                        break 
            else:
                if score < best_score: # store min score
                    best_score = score
                    best_move = move
                if alpha_beta:
                    beta = min(beta, best_score)
                    if beta <= alpha: # similarly, we can prune branches from a max descendant that has a alpha >= n (beta)
                        break  

        return best_score, best_move, time.time() - start_time # will return the time for action

    """
    Method that plays a turn as an AI:
    Args:
        - file: output file for game. 
    Returns:
        - move: move being played by AI
    """
    def play_turn_AI(self, file, max_depth):
        best_score, move, time_for_action = self.minimax(self.current_game_state, max_depth, 0, time.time(), alpha_beta=self.alpha_beta)
        
        # check if move is valid, if not, AI loses
        if not self.is_valid_move(self.current_game_state, move):
            print("AI made an invalid move. AI loses.")
            file.write("\nAI made an invalid move. AI loses.\n")
            exit(1)
        
        return best_score, move, time_for_action
    
    """
    Logic to play a turn as a human.
    Args:
        - file: output file for game.
    Returns:
        - move: move being played by human.
    """
    def play_turn_H(self, file):
        player = self.current_game_state["turn"]
        while(self.warnings[player] < 2):
            move = input(f"{self.current_game_state['turn'].capitalize()} to move: ")
            if move.lower() == 'exit':
                print("Game exited.")
                exit(1)

            move = self.parse_input(move)

            if not move or not self.is_valid_move(self.current_game_state, move):
                self.warnings[player] += 1
                print(f"Invalid move! Warning: {player} has 1 warning. Next invalid move will result in a loss.")
                continue

            # If we reach here move is valid and we reset warnings
            player = self.current_game_state["turn"]
            self.warnings[player] = 0
            return move
        
        # if we reach here, player has made 2 invalid moves and loses
        print(f"Invalid move! {player} loses the game due to repeated rule violations.")
        file.write(f"\n{player.capitalize()} loses the game due to repeated rule violations.\n")
        exit(1)


    """
    Game loop     
    Args:
        - None
    Returns:
        - None
    """
    def play(self):

        # Ensure output dir exists and create full output file name
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_filename = os.path.join(self.output_dir, self.output_filename)

        with open(self.output_filename, 'w') as f:
            # Write game parameters
            f.write(f"Timeout: {self.timeout} seconds\n")
            f.write(f"Max turns: {self.max_turns}\n")
            f.write(f"Play mode: {self.play_mode}\n")
            f.write(f"Alpha-beta: {self.alpha_beta}\n")
            if self.play_mode == 'H-H':
                f.write("Heuristic: None\n\n")
            else:
                f.write(f"Heuristic: {self.heuristic.__name__}\n\n")
            f.write("Initial board configuration:\n")
            f.write(self.board_to_string())

            # Set players
            mode = self.play_mode.split('-')
            white = mode[0]
            black = mode[1]

            print("Welcome to Mini Chess! Enter moves as 'B2 B3'. Type 'exit' to quit.")
            while True:
                # draw condition
                if self.current_game_state["move_count"] >= self.max_turns:
                    print("Maximum turns reached. Game is a draw!")
                    f.write("\nMaximum turns reached. Game is a draw!\n")
                    break

                self.display_board()

                # Set other vars
                search_score = None
                time_for_action = None

                # Check who's move it is and play accordingly:
                if self.current_game_state["turn"] == "white":
                    if white == 'H':
                        move = self.play_turn_H(f)
                    else:
                        search_score, move, time_for_action = self.play_turn_AI(f, 8)
                else:
                    if black == 'H':
                        move = self.play_turn_H(f)
                    else:
                        search_score, move, time_for_action = self.play_turn_AI(f, 8)
                
                # update move count and turn number
                current_move_count = self.current_game_state["move_count"]
                turn_number = (current_move_count // 2) + 1
                print(f"Turn #{turn_number}")

                start, end = move
                start_coord = self.convert_to_notation(start)
                end_coord = self.convert_to_notation(end)
                player = self.current_game_state['turn']

                f.write(f"\nPlayer: {player}\n")
                f.write(f"Turn #: {turn_number}\n")
                f.write(f"Action: move from {start_coord} to {end_coord}\n")

                # Add AI Statistics
                if (search_score is not None and time_for_action is not None):
                    self._print_ai_stats(f, search_score, time_for_action)

                self.make_move(self.current_game_state, move)

                f.write("\nNew board configuration:\n")
                f.write(self.board_to_string())

                if self.is_king_captured(self.current_game_state):
                    winner = "black" if player == "white" else "white"
                    total_turns = (self.current_game_state["move_count"] // 2)
                    f.write(f"\n{winner.capitalize()} won in {total_turns} turns\n")
                    print(f"\n{winner.capitalize()} won in {total_turns} turns")
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mini Chess Game')
    parser.add_argument('--alpha-beta', type=bool, default=False, help='Enable alpha-beta pruning (True/False)')
    parser.add_argument('--timeout', type=float, default=2, help='Timeout per move (seconds)')
    parser.add_argument('--max_turns', type=int, default=100, help='Maximum number of turns')
    parser.add_argument('--play-mode', type=str, default='H-H', help='Play mode (H-H, H-AI, AI-H, AI-AI)')
    parser.add_argument('--heuristic', type=str, default='None', help='e0, e1 or e2')
    args = parser.parse_args()

    # Set heuristic function
    heuristic = None
    if args.heuristic == 'e0':
        heuristic = heuristic_e0
    elif args.heuristic == 'e1':
        heuristic = heuristic_e1
    elif args.heuristic == 'e2':
        heuristic = heuristic_e2
    elif args.heuristic == 'None':
        heuristic = 'None'
    else:
        print("Heuristic not recognized, using e0...")
        heuristic = heuristic_e0

    game = MiniChess(args.alpha_beta, args.timeout, args.max_turns, args.play_mode, heuristic)
    game.play()  