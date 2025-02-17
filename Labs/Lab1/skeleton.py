import gym
import random
import requests
import numpy as np
import argparse
import sys
from gym_connect_four import ConnectFourEnv

envCopy: ConnectFourEnv = gym.make("ConnectFour-v0")  # type: ignore
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
    print("currentPlayer", env.get_current_player())
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

    return state, reward, done


def student_move(env):
    env.change_player()  # change back to student
    unchangedBoard = env.get_state()
    print("CurrentPlayer: ", env.get_current_player())
    print("Current state:\n", env.get_state())

    moves = env.available_moves()

    print("Available moves: ", moves)
    move, _ = minimax(env, 3, -np.inf, np.inf, True)
    print("Move: ", move)
    env.reset(board=unchangedBoard)
    return move


def evaluate_board(env):
    if env.get_current_player == 1:
        return 1
    else:
        return -1


def minimax(env, depth, alpha, beta, maximizingPlayer):
    envCopy = env
    print(
        "board:\n",
        envCopy.get_state(),
        "depth: ",
        depth,
        "alpha: ",
        alpha,
        "beta: ",
        beta,
        "maximizingPlayer: ",
        maximizingPlayer,
    )
    if depth == 0:
        return None, evaluate_board(envCopy)
    if envCopy.is_win_state():
        return None, evaluate_board(envCopy)

    if maximizingPlayer:
        maxScore = -np.inf
        if envCopy.get_current_player() != 1:
            envCopy.change_player()
        for move in envCopy.available_moves():
            curColumn = move
            envCopy.step(move)
            # print("Maximizing move: ", move)
            _, score = minimax(envCopy, depth - 1, alpha, beta, False)
            if score > maxScore:
                maxScore = score
                curColumn = move
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        return curColumn, maxScore

    else:
        minScore = np.inf
        for move in envCopy.available_moves():
            curColumn = move
            envCopy.change_player()
            envCopy.step(move)
            _, score = minimax(envCopy, depth - 1, alpha, beta, True)
            if score < minScore:
                minScore = score
            beta = min(beta, score)
            if beta <= alpha:
                break
        return curColumn, minScore


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
        envCopy.reset(board=state)
    else:
        # reset game to starting state
        envCopy.reset(board=None)
        # determine first player
        student_gets_move = random.choice([True, False])
        if student_gets_move:
            print("\nYou start!\n")
        else:
            print("\nBot start!\n")

    # Print current gamestate
    print("Current state (1 are student discs, -1 are servers, 0 is empty): ")
    print(state)

    done = False
    while not done:
        # Select your move
        stmove = student_move(envCopy)  # TODO: change input here

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
            envCopy.reset(board=state)
        else:
            if student_gets_move:  # type: ignore
                # Execute your move
                avmoves = envCopy.available_moves()
                if stmove not in avmoves:
                    print("You tried to make an illegal move! You have lost the game.")
                    break
                else:
                    state, result, done, _ = envCopy.step(stmove)  # type: ignore || TODO: is this an error?
            student_gets_move = True  # student only skips move first turn if bot starts

            # print or render state here if you like

            # select and make a move for the opponent, returned reward from students view
            if not done:
                state, result, done = opponents_move(envCopy)  # type: ignore
        # Check if the game is over
        if result != 0:  # type: ignore
            done = True
            if not vs_server:
                print("Game over. ", end="")
            if result == 1:  # type: ignore
                print("You won!")
            elif result == 0.5:  # type: ignore
                print("It's a draw!")
            elif result == -1:  # type: ignore
                print("You lost!")
            elif result == -10:  # type: ignore
                print("You made an illegal move and have lost!")
            else:
                print("Unexpected result result={}".format(result))  # type: ignore
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
