from agent import Agent
from oxono import Game, State
import time


class MyAgent(Agent):
    def __init__(self, player):
        super().__init__(player)

    def act(self, state: State, remaining_time):
        nb_pieces = 32 - state.pieces_x[0] - state.pieces_x[1] - state.pieces_o[0] - state.pieces_o[1]

        timout = max((remaining_time / max(nb_pieces, 1)) * 0.8, 0.5)
        start = time.time()

        value, move = max_value(
            state=state,
            alpha=float("-inf"),
            beta=float("inf"),
            cutoff=calculer_depth(nb_pieces),
            root_player=self.player,
            timout=timout,
            start=start
        )
        
        if move is None:
            actions = Game.actions(state)
            return actions[0] if actions else None
        return move
    

def calculer_depth(nb_pieces : int):
    if nb_pieces < 10:
        return 5
    elif nb_pieces < 24:
        return 6
    else:
        return 6 + (nb_pieces - 24) // 3


def max_value(state: State, alpha, beta, cutoff: int, root_player: int, timout : float, start : float):
    if Game.is_terminal(state):
        return Game.utility(state, root_player), None
    
    now = time.time() - start
    if cutoff == 0 or now >= timout:
        return evaluation(state, root_player), None

    v = float("-inf")
    best_move = None

    for a in Game.actions(state):
        copy_state = state.copy()
        Game.apply(copy_state, a)

        v2, _ = min_value(copy_state, alpha, beta, cutoff - 1, root_player, timout, start)

        if v2 > v:
            v = v2
            best_move = a

        alpha = max(alpha, v)
        if alpha >= beta:
            break

    return v, best_move


def min_value(state: State, alpha, beta, cutoff: int, root_player: int, timout : float, start : float):
    if Game.is_terminal(state):
        return Game.utility(state, root_player), None
    
    now = time.time() - start
    if cutoff == 0 or now >= timout:
        return evaluation(state, root_player), None

    v = float("inf")
    best_move = None

    for a in Game.actions(state):
        copy_state = state.copy()
        Game.apply(copy_state, a)

        v2, _ = max_value(copy_state, alpha, beta, cutoff - 1, root_player, timout, start)

        if v2 < v:
            v = v2
            best_move = a

        beta = min(beta, v)
        if alpha >= beta:
            break

    return v, best_move


def evaluation(state: State, root_player: int):
    if Game.is_terminal(state): return Game.utility(state, root_player)

    opponent = 1 - root_player
    board = state.board

    score = 0
    
    for i in range(6):
        # lignes
        score += line_score(board[i], root_player, opponent)
        
        # colonnes
        score += line_score([board[j][i] for j in range(6)], root_player, opponent)

    # Petit bonus
    own_pieces = state.pieces_o[root_player] + state.pieces_x[root_player]
    opponent_pieces = state.pieces_o[opponent] + state.pieces_x[opponent]
    score += 0.5 * (own_pieces - opponent_pieces)

    return score


def line_score(cells, root_player : int, opponent : int):
    own = 0
    for c in cells :
        if c is not None and c[1] == root_player : own += 1
        
    adv = 0
    for c in cells :
        if c is not None and c[1] == opponent : adv += 1

    empty = 0
    for c in cells :
        if c is None : empty += 1
        
    if own > 0 and adv > 0:
        return 0
    
    score = 0
    
    if own == 3 and empty >= 1:
        score += 40
    elif own == 2 and empty >= 2:
        score += 10
    elif own == 1 and empty >= 3:
        score += 2
        
    if adv == 3 and empty >= 1:
        score -= 50
    elif adv == 2 and empty >= 2:
        score -= 12
    elif adv == 1 and empty >= 3:
        score -= 2

    return score