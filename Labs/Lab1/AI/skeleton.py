import gym
import random
import requests
import numpy as np
import argparse
import sys
from gym_connect_four import ConnectFourEnv

env: ConnectFourEnv = gym.make("ConnectFour-v0")
sys.setrecursionlimit(99999999)

SERVER_ADDRESS = "https://vilde.cs.lth.se/edap01-4inarow/"
API_KEY = "nyckel"
STIL_ID = ["ma7230al-s"]


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
    env.change_player()  # change to opponent
    avmoves = env.available_moves()
    print("Available moves: ", avmoves)
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


def student_move(env):
    # Get available moves
    avmoves = env.available_moves()
    best_move = None
    best_score = float("-inf")  # Initialize best score to negative infinity
    alpha = float("-inf")  # Initialize alpha to negative infinity
    beta = float("inf")  # Initialize beta to positive infinity

    # Loop through available moves
    for move in avmoves:
        # Apply the move to a copy of the current environment state
        next_state, _, _ = env.clone().step(move)
        # Calculate the score for the move using minimax with alpha-beta pruning
        score = alpha_beta(
            next_state, False, alpha, beta, depth=3
        )  # Adjust depth as needed
        # Update alpha with the maximum of alpha and the current score
        alpha = max(alpha, score)
        # If the score for this move is better than the current best score, update best score and best move
        if score > best_score:
            best_score = score
            best_move = move

    # Return the best move found
    return best_move


def alpha_beta(state, maximizing_player, alpha, beta, depth):
    # Check if the game is over or depth limit reached
    if depth == 0 or state.is_terminal():
        return evaluate_state(state)

    # Get available moves
    avmoves = state.available_moves()

    if maximizing_player:
        value = float("-inf")
        for move in avmoves:
            # Apply the move to a copy of the current state
            next_state, _, _ = state.clone().step(move)
            # Recursively call alpha_beta with the next state and switch players
            value = max(value, alpha_beta(next_state, False, alpha, beta, depth - 1))
            alpha = max(alpha, value)
            # Perform alpha-beta pruning
            if beta <= alpha:
                break
        return value
    else:
        value = float("inf")
        for move in avmoves:
            next_state, _, _ = state.clone().step(move)
            value = min(value, alpha_beta(next_state, True, alpha, beta, depth - 1))
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value


def evaluate_state(state):
    # Define weights for different features
    weights = {
        "player_1": 1,  # Your discs
        "player_2": -1,  # Opponent's discs
        "empty": 0.1,  # Empty spaces
        "win": 1000,  # Winning state
        "lose": -1000,  # Losing state
        "draw": 0,  # Draw state
    }

    # Check if the game is over
    if state.is_win():
        return weights["win"]
    elif state.is_draw():
        return weights["draw"]
    elif state.is_loss():
        return weights["lose"]

    # Initialize evaluation score
    score = 0

    # Get the board state
    board = state.board

    # Evaluate each position on the board
    for row in range(state.rows):
        for col in range(state.cols):
            if board[row][col] == 1:
                # Add score for player 1 (your discs)
                score += weights["player_1"]
            elif board[row][col] == -1:
                # Subtract score for player 2 (opponent's discs)
                score += weights["player_2"]
            else:
                # Add small score for empty spaces
                score += weights["empty"]

    return score


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

    # default state 6 rad 7 col
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
            print("Bot start!")
            print()

    # Print current gamestate
    print("Current state (1 are student discs, -1 are servers, 0 is empty): ")
    print(state)
    print()

    done = False
    while not done:
        # Select your move
        stmove = student_move(env)  # TODO: change input here

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
                    print("You tried to make an illegal move! You have lost the game.")
                    break
                state, result, done, _ = env.step(stmove)

            student_gets_move = True  # student only skips move first turn if bot starts

            # print or render state here if you like
            print("You placed a disc in column ", stmove)

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
