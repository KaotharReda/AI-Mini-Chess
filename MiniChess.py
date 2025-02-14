import math
import copy
import time
import argparse

class MiniChess:
    def __init__(self, alpha_beta, timeout, max_turns, play_mode):
        self.current_game_state = self.init_board()
        self.alpha_beta = alpha_beta
        self.timeout = timeout
        self.max_turns = max_turns
        self.play_mode = play_mode
        self.output_filename = f"gameTrace.txt"
        self.warnings = {"white": 0, "black": 0}


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
        - game_state: Dictionary representing the current game state
    Returns:
        - None
    """
    def display_board(self, game_state):
        print()
        for i, row in enumerate(game_state["board"], start=1):
            print(str(6-i) + "  " + ' '.join(piece.rjust(3) for piece in row))
        print()
        print("     A   B   C   D   E")
        print()


    def board_to_string(self, game_state):
        board_str = ""
        for i, row in enumerate(game_state["board"], start=1):
            board_str += str(6 - i) + "  " + ' '.join(piece.rjust(3) for piece in row) + "\n"
        board_str += "\n     A   B   C   D   E\n\n"
        return board_str


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
        return True

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
        for s_row in range(5):
            for s_col in range(5):
                piece = game_state["board"][s_row][s_col]
                if piece == '.' or piece[0] != game_state["turn"][0]:
                    continue

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
                                    valid_moves.append(((s_row, s_col), (e_row, e_col)))

                elif piece[1] == 'Q':  # Queen
                    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
                    for dr, dc in directions:
                        e_row, e_col = s_row + dr, s_col + dc
                        while 0 <= e_row < 5 and 0 <= e_col < 5:
                            target = game_state["board"][e_row][e_col]
                            if target == '.' or target[0] != piece[0]:
                                valid_moves.append(((s_row, s_col), (e_row, e_col)))
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

                elif piece[1] == 'p':  # Pawn
                    direction = -1 if piece[0] == 'w' else 1
                    # Move forward
                    e_row, e_col = s_row + direction, s_col
                    if 0 <= e_row < 5 and game_state["board"][e_row][e_col] == '.':
                        valid_moves.append(((s_row, s_col), (e_row, e_col)))
                    # Capture diagonally
                    for dc in [-1, 1]:
                        e_row, e_col = s_row + direction, s_col + dc
                        if 0 <= e_row < 5 and 0 <= e_col < 5:
                            target = game_state["board"][e_row][e_col]
                            if target != '.' and target[0] != piece[0]:
                                valid_moves.append(((s_row, s_col), (e_row, e_col)))

        return valid_moves
    """
    Modify to board to make a move

    Args: 
        - game_state:   dictionary | Dictionary representing the current game state
        - move          tuple | the move to perform ((start_row, start_col),(end_row, end_col))
    Returns:
        - game_state:   dictionary | Dictionary representing the modified game state
    """
    def make_move(self, game_state, move):
        start = move[0]
        end = move[1]
        start_row, start_col = start
        end_row, end_col = end
        piece = game_state["board"][start_row][start_col]
        game_state["board"][start_row][start_col] = '.'
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

    def is_king_captured(self, game_state):
        current_turn = game_state["turn"]
        opponent = "black" if current_turn == "white" else "white"
        king = f"{opponent[0]}K"
        for row in game_state["board"]:
            if king in row:
                return False
        return True

    """
    Game loop

    Args:
        - None
    Returns:
        - None
    """
    def play(self):
        with open(self.output_filename, 'w') as f:
            # Write game parameters
            f.write(f"Timeout: {self.timeout} seconds\n")
            f.write(f"Max turns: {self.max_turns}\n")
            f.write(f"Play mode: {self.play_mode}\n")
            f.write(f"Alpha-beta: {self.alpha_beta}\n\n")
            f.write("Initial board configuration:\n")
            f.write(self.board_to_string(self.current_game_state))


            print("Welcome to Mini Chess! Enter moves as 'B2 B3'. Type 'exit' to quit.")
            while True:
                self.display_board(self.current_game_state)
                move = input(f"{self.current_game_state['turn'].capitalize()} to move: ")
                if move.lower() == 'exit':
                    print("Game exited.")
                    exit(1)

                move = self.parse_input(move)
                if not move or not self.is_valid_move(self.current_game_state, move):
                    print("Invalid move. Try again.")
                    continue


                valid_moves = self.valid_moves(self.current_game_state)
                if move not in valid_moves:
                    player = self.current_game_state["turn"]
                    self.warnings[player] += 1
                    if self.warnings[player] == 1:
                        print(f"Invalid move! Warning: {player} has 1 warning. Next invalid move will result in a loss.")
                    elif self.warnings[player] >= 2:
                        print(f"Invalid move! {player} loses the game due to repeated rule violations.")
                        f.write(f"\n{player.capitalize()} loses the game due to repeated rule violations.\n")
                        print(f"\n{player.capitalize()} loses the game due to repeated rule violations.")
                        break
                    continue

                # Reset warnings if the move is valid
                player = self.current_game_state["turn"]
                self.warnings[player] = 0

                current_move_count = self.current_game_state["move_count"]
                turn_number = (current_move_count // 2) + 1
                start, end = move
                start_coord = self.convert_to_notation(start)
                end_coord = self.convert_to_notation(end)
                player = self.current_game_state['turn']

                f.write(f"\nPlayer: {player}\n")
                f.write(f"Turn #: {turn_number}\n")
                f.write(f"Action: move from {start_coord} to {end_coord}\n")

                self.make_move(self.current_game_state, move)

                f.write("New board configuration:\n")
                f.write(self.board_to_string(self.current_game_state))

                if self.is_king_captured(self.current_game_state):
                    winner = "black" if player == "white" else "white"
                    total_turns = (self.current_game_state["move_count"] // 2) + 1
                    f.write(f"\n{winner.capitalize()} wins in {total_turns} turns!\n")
                    print(f"\n{winner.capitalize()} wins in {total_turns} turns!")
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mini Chess Game')
    parser.add_argument('--alpha_beta', type=bool, default=False, help='Enable alpha-beta pruning (True/False)')
    parser.add_argument('--timeout', type=int, default=5, help='Timeout per move (seconds)')
    parser.add_argument('--max_turns', type=int, default=100, help='Maximum number of turns')
    parser.add_argument('--play_mode', type=str, default='H-H', help='Play mode (H-H, H-AI, AI-H, AI-AI)')
    args = parser.parse_args()

    game = MiniChess(args.alpha_beta, args.timeout, args.max_turns, args.play_mode)
    game.play()