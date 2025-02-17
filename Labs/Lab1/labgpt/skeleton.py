# NOTE: This is the code from ChatGPT 4. ma7230al-s
# NOTE: This is the code from ChatGPT 4. ma7230al-s
# NOTE: This is the code from ChatGPT 4. ma7230al-s
# NOTE: This is the code from ChatGPT 4. ma7230al-s
# NOTE: This is the code from ChatGPT 4. ma7230al-s
# NOTE: This is the code from ChatGPT 4. ma7230al-s
# NOTE: This is the code from ChatGPT 4. ma7230al-s
# NOTE: This is the code from ChatGPT 4. ma7230al-s
# NOTE: This is the code from ChatGPT 4. ma7230al-s
# NOTE: This is the code from ChatGPT 4. ma7230al-s
# NOTE: This is the code from ChatGPT 4. ma7230al-s
# NOTE: This is the code from ChatGPT 4. ma7230al-s
# NOTE: This is the code from ChatGPT 4. ma7230al-s
# NOTE: This is the code from ChatGPT 4. ma7230al-s
# NOTE: This is the code from ChatGPT 4. ma7230al-s
# NOTE: This is the code from ChatGPT 4. ma7230al-s
# NOTE: This is the code from ChatGPT 4. ma7230al-s
# NOTE: This is the code from ChatGPT 4. ma7230al-s
import copy
import gym
import random
import requests
import numpy as np
import argparse
import sys
from gym_connect_four import ConnectFourEnv

env: ConnectFourEnv = gym.make(
    "ConnectFour-v0"
)  # NOTE: This is the code from ChatGPT 4. ma7230al-s

SERVER_ADDRESS = "https://vilde.cs.lth.se/edap01-4inarow/"
API_KEY = "nyckel"
STIL_ID = ["da20example-s1", "da22test-s2"]


def call_server(move):
    res = requests.post(
        SERVER_ADDRESS + "move",
        data={
            "stil_id": STIL_ID,
            "move": move,  # -1 signals the system to start a new game. any running game is counted as a loss
            "api_key": API_KEY,
        },
    )
    # For safety some respose checking is done here
    if res.status_code != 200:
        print("Server gave a bad response, error code={}".format(res.status_code))
        exit()
    if not res.json()["status"]:
        print("Server returned a bad status. Return message: ")
        print(res.json()["msg"])
        exit()
    return res


def check_stats():
    res = requests.post(
        SERVER_ADDRESS + "stats",
        data={
            "stil_id": STIL_ID,
            "api_key": API_KEY,
        },
    )

    stats = res.json()
    return stats


"""
You can make your code work against this simple random agent
before playing against the server.
It returns a move 0-6 or -1 if it could not make a move.
To check your code for better performance, change this code to
use your own algorithm for selecting actions too
"""


def opponents_move(env):
    env.change_player()  # change to oppoent
    avmoves = env.available_moves()
    if not avmoves:
        env.change_player()  # change back to student before returning
        return -1

    # TODO: Optional? change this to select actions with your policy too
    # that way you get way more interesting games, and you can see if starting
    # is enough to guarrantee a win
    action = random.choice(list(avmoves))

    state, reward, done, _ = env.step(action)
    if done:
        if reward == 1:  # reward is always in current players view
            reward = -1
    env.change_player()  # change back to student before returning
    return state, reward, done


def student_move():
    _, best_move = minimax_alpha_beta(
        env, depth=4, alpha=float("-inf"), beta=float("inf"), maximizing_player=True
    )
    return best_move


def minimax_alpha_beta(env, depth, alpha, beta, maximizing_player):
    if depth == 0 or env.is_win_state():  # NOTE: Changed
        return (
            evaluate_score(env.board),
            None,
        )  # NOTE: Changed removed the - from evaluate_score

    valid_moves = env.available_moves()
    # Convert the frozenset to a list and then select the first element
    if valid_moves:  # Ensure there are available moves
        best_move = list(valid_moves)[0]
    else:
        best_move = (
            None  # or any default invalid move indication if no moves are available
        )

    if maximizing_player:
        max_eval = float("-inf")
        for move in valid_moves:
            newEnv = copy.deepcopy(env)  # NOTE: Changed
            newEnv.step(move)  # NOTE: Changed
            eval, _ = minimax_alpha_beta(newEnv, depth - 1, alpha, beta, False)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float("inf")
        for move in valid_moves:
            newEnv = copy.deepcopy(env)  # NOTE: Changed
            newEnv.step(move)  # NOTE: Changed
            eval, _ = minimax_alpha_beta(newEnv, depth - 1, alpha, beta, True)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move


def evaluate_score(board):
    """
    Evaluate the board state from the perspective of the player (1).

    Args:
    - board: The current state of the board, a 2D numpy array.

    Returns:
    - score: The evaluated score of the board.
    """
    score = 0

    # Scoring values
    SCORES = {
        "four": 100,
        "open_three": 10,
        "open_two": 5,
    }

    # Directions to check: horizontal, vertical, diagonal (/ and \)
    DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]

    ROWS, COLS = board.shape
    for row in range(ROWS):
        for col in range(COLS):
            if board[row][col] == 1:  # Player's disc
                for direction in DIRECTIONS:
                    score += check_direction(board, row, col, direction, SCORES)

            elif board[row][col] == -1:  # Opponent's disc
                for direction in DIRECTIONS:
                    score -= check_direction(board, row, col, direction, SCORES)

    return score


def check_direction(board, row, col, direction, scores):
    """
    Check a single direction from a starting point for scoring patterns.

    Args:
    - board: The board state.
    - row, col: Starting point coordinates.
    - direction: The direction to check.
    - scores: A dictionary of scoring values for different patterns.

    Returns:
    - dir_score: The score for patterns found in this direction.
    """
    dir_score = 0
    ROWS, COLS = board.shape
    dr, dc = direction

    # Counters for discs in a sequence
    player_count = 0

    for i in range(1, 4):  # Check up to three spaces in the given direction
        new_row, new_col = row + dr * i, col + dc * i
        if 0 <= new_row < ROWS and 0 <= new_col < COLS:
            if board[new_row][new_col] == 1:  # Player's disc
                player_count += 1
            else:
                break  # Stop if the sequence is broken
        else:
            break  # Out of bounds

    # Adjust score based on the counters
    if player_count == 3:
        dir_score += scores["four"]
    elif player_count == 2:
        dir_score += scores["open_three"]
    elif player_count == 1:
        dir_score += scores["open_two"]

    return dir_score


def play_game(vs_server=False):
    """
    The reward for a game is as follows. You get a
    botaction = random.choice(list(avmoves)) reward from the
    server after each move, but it is 0 while the game is running
    loss = -1
    win = +1
    draw = +0.5
    error = -10 (you get this if you try to play in a full column)
    Currently the player always makes the first move
    """

    # default state
    state = np.zeros((6, 7), dtype=int)

    # setup new game
    if vs_server:
        # Start a new game
        res = call_server(
            -1
        )  # -1 signals the system to start a new game. any running game is counted as a loss

        # This should tell you if you or the bot starts
        print(res.json()["msg"])
        botmove = res.json()["botmove"]
        state = np.array(res.json()["state"])
        # reset env to state from the server (if you want to use it to keep track)
        env.reset(board=state)
    else:
        # reset game to starting state
        env.reset(board=None)
        # determine first player
        student_gets_move = random.choice([True, False])
        if student_gets_move:
            print("You start!")
            print()
        else:
            print("Bot starts!")
            print()

    # Print current gamestate
    print("Current state (1 are student discs, -1 are servers, 0 is empty): ")
    print(state)
    print()

    done = False
    while not done:
        # Select your move
        stmove = student_move()  # TODO: change input here

        # make both student and bot/server moves
        if vs_server:
            # Send your move to server and get response
            res = call_server(stmove)
            print(res.json()["msg"])

            # Extract response values
            result = res.json()["result"]
            botmove = res.json()["botmove"]
            state = np.array(res.json()["state"])
            # reset env to state from the server (if you want to use it to keep track)
            env.reset(board=state)
        else:
            if student_gets_move:
                # Execute your move
                avmoves = env.available_moves()
                if stmove not in avmoves:
                    print("You tied to make an illegal move! You have lost the game.")
                    break
                state, result, done, _ = env.step(stmove)

            student_gets_move = True  # student only skips move first turn if bot starts

            # print or render state here if you like

            # select and make a move for the opponent, returned reward from students view
            if not done:
                state, result, done = opponents_move(env)

        # Check if the game is over
        if result != 0:
            done = True
            if not vs_server:
                print("Game over. ", end="")
            if result == 1:
                print("You won!")
            elif result == 0.5:
                print("It's a draw!")
            elif result == -1:
                print("You lost!")
            elif result == -10:
                print("You made an illegal move and have lost!")
            else:
                print("Unexpected result result={}".format(result))
            if not vs_server:
                print("Final state (1 are student discs, -1 are servers, 0 is empty): ")
        else:
            print("Current state (1 are student discs, -1 are servers, 0 is empty): ")

        # Print current gamestate
        print(state)
        print()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-l", "--local", help="Play locally", action="store_true")
    group.add_argument(
        "-o", "--online", help="Play online vs server", action="store_true"
    )
    parser.add_argument(
        "-s", "--stats", help="Show your current online stats", action="store_true"
    )
    args = parser.parse_args()

    # Print usage info if no arguments are given
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.local:
        play_game(vs_server=False)
    elif args.online:
        play_game(vs_server=True)

    if args.stats:
        stats = check_stats()
        print(stats)

    # TODO: Run program with "--online" when you are ready to play against the server
    # the results of your games there will be logged
    # you can check your stats bu running the program with "--stats"


if __name__ == "__main__":
    main()
