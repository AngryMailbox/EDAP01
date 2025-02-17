def evaluate_position(env, row, col, player):
    score = 0

    # Horizontal
    for i in range(4):
        if col + i < 7:
            if env.board[row][col + i] == 1:
                score += 1
            else:
                break
    for i in range(4):
        if col - i >= 0:
            if env.board[row][col - i] == 1:
                score += 1
            else:
                break

    # Vertical
    for i in range(4):
        if row + i < 6:
            if env.board[row + i][col] == 1:
                score += 1
            else:
                break

    # Diagonal
    for i in range(4):
        if row + i < 6 and col + i < 7:
            if env.board[row + i][col + i] == 1:
                score += 1
            else:
                break
    for i in range(4):
        if row - i >= 0 and col - i >= 0:
            if env.board[row - i][col - i] == 1:
                score += 1
            else:
                break

    return score


def evaluate_board(env):
    score = 0
    for row in range(6):
        for col in range(7):
            if env.board[row][col] == 1:
                score += evaluate_position(env, row, col, 1)
            elif env.board[row][col] == -1:
                score -= evaluate_position(env, row, col, -1)
    return score


def minimax(board, depth, alpha, beta, maximizingPlayer):
    if depth == 0 or game is won?

    if maximizingPlayer:
        maxEval = -np.inf
        for move in env.available_moves():
            eval = minimax(board, depth - 1, alpha, beta, False)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return maxEval

    else:
        minEval = np.inf
        for move in env.available_moves():
            eval = minimax(board, depth - 1, alpha, beta, True)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return minEval