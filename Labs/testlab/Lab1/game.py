import gym
import random
import requests
import numpy as np
import argparse
import sys
from gym_connect_four import ConnectFourEnv
import copy

env: ConnectFourEnv = gym.make("ConnectFour-v0")  # type: ignore
sys.setrecursionlimit(999999999)

SERVER_ADDRESS = "https://vilde.cs.lth.se/edap01-4inarow/"
API_KEY = "nyckel"
STIL_ID = ["ma7230al-s"]

score_matrix = np.array(
    [
        [2, 3, 5, 6, 5, 3, 2],
        [3, 4, 8, 10, 8, 4, 3],
        [4, 5, 11, 15, 11, 5, 4],
        [4, 5, 11, 15, 11, 5, 4],
        [3, 4, 8, 10, 8, 4, 3],
        [4, 6, 7, 8, 7, 6, 4],
    ]
)


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
    env.change_player()
    avmoves = list(env.available_moves())
    if not avmoves:
        env.change_player()
        return -1

    action = random.choice(list(avmoves))

    state, reward, done, _ = env.step(action)
    if done:
        if reward == 1:  # 1 here means the student did not win.
            reward = -1

    env.change_player()
    return state, reward, done


def minimax(pos, depth, alpha, beta, isMaximizingPlayer):
    if depth == 0 or pos.is_win_state():
        return evaluate(pos.board), None  # No more moves

    moves = list(pos.available_moves())

    newMove = None
    if isMaximizingPlayer:
        maxVal = -np.inf
        for move in moves:
            newPos = copy.deepcopy(pos)
            newPos.step(move)
            val, _ = minimax(newPos, depth - 1, alpha, beta, False)
            if val > maxVal:
                maxVal = val
                newMove = move
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        return maxVal, newMove
    else:
        minVal = np.inf
        for move in moves:
            newPos = copy.deepcopy(pos)
            newPos.step(move)
            val, _ = minimax(newPos, depth - 1, alpha, beta, True)
            if val < minVal:
                minVal = val
                newMove = move
            beta = min(beta, val)
            if alpha >= beta:
                break
        return minVal, newMove


def evaluate(board):
    score = 0
    if checkWin(board, 1):
        score = np.inf
    elif checkWin(board, -1):
        score = -np.inf

    for i in range(6):
        for j in range(7):
            if board[i][j] == 1:
                score += score_matrix[i][j]
            elif board[i][j] == -1:
                score -= score_matrix[i][j]
    return score


def checkWin(board, player):
    rows, cols = board.shape

    # horizontal
    for c in range(cols - 3):
        for r in range(rows):
            if (
                board[r, c] == player
                and board[r, c + 1] == player
                and board[r, c + 2] == player
                and board[r, c + 3] == player
            ):
                return True

    # vertical
    for c in range(cols):
        for r in range(rows - 3):
            if (
                board[r, c] == player
                and board[r + 1, c] == player
                and board[r + 2, c] == player
                and board[r + 3, c] == player
            ):
                return True

    # Bottom left to top right
    for c in range(cols - 3):
        for r in range(rows - 3):
            if (
                board[r, c] == player
                and board[r + 1, c + 1] == player
                and board[r + 2, c + 2] == player
                and board[r + 3, c + 3] == player
            ):
                return True

    # Top right to bottom left
    for c in range(cols - 3):
        for r in range(3, rows):
            if (
                board[r, c] == player
                and board[r - 1, c + 1] == player
                and board[r - 2, c + 2] == player
                and board[r - 3, c + 3] == player
            ):
                return True
    return False


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
        # print(res)
        # print(res.json()["msg"]) # This should tell you if you or the bot starts
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
        else:
            print("Bot start!")

    # Print current gamestate
    print("Current state (1 are student discs, -1 are servers, 0 is empty): ")
    print(state)

    done = False
    while not done:
        # Select your move
        _, stmove = minimax(env, 4, -np.inf, np.inf, True)

        if vs_server:
            res = call_server(stmove)
            print(res.json()["msg"])

            result = res.json()["result"]
            botmove = res.json()["botmove"]
            state = np.array(res.json()["state"])
            env.reset(board=state)
        else:
            if student_gets_move:  # type: ignore
                avmoves = list(env.available_moves())
                if stmove not in avmoves:
                    print("You tried to make an illegal move! You have lost the game.")
                    break
                else:
                    state, result, done, _ = env.step(stmove)  # type: ignore
            student_gets_move = True  # student only skips move first turn if bot starts

            # select and make a move for the opponent, returned reward from students view
            if not done:
                state, result, done = opponents_move(env)  # type: ignore
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
            pass

        # Prints current gamestate a bit nicer. X is student, O is server
        print(
            np.array(
                [
                    ["X" if x == 1 else "O" if x == -1 else " " for x in row]
                    for row in state
                ]
            )
        )
        # print(state)
        # print()


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


if __name__ == "__main__":
    main()
