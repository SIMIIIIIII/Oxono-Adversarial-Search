from agent import Agent
from oxono import Game, State
import time
import random


class MyAgent(Agent):
    def __init__(self, player):
        super().__init__(player)

    def act(self, state: State, remaining_time):

        timeout =time.perf_counter() + get_budget(state, remaining_time, self.player)
        cutoff = get_depth(state)

        if cutoff == 0 : return random.choice(Game.actions(state))

        value, move = max_value(
            state=state,
            alpha=float("-inf"),
            beta=float("inf"),
            cutoff=cutoff,
            root_player=self.player,
            timeout=timeout,
        )
        
        if move is None:
            actions = Game.actions(state)
            return actions[0] if actions else None
        return move
    

def get_budget(state: State, remaining_time: float, player: int) -> float:
    remaining_moves = state.pieces_x[player] + state.pieces_o[player]
    safe_remaining = max(remaining_time - 0.2, 0.05)

    if remaining_moves <= 1:
        return safe_remaining

    nominal = safe_remaining / remaining_moves
    return max(0.05, min(safe_remaining, nominal * 1.1))
    

def get_depth(state: State) -> int:
    pieces_in_hand = (
        state.pieces_x[0] + state.pieces_x[1] +
        state.pieces_o[0] + state.pieces_o[1]
    )
    
    if pieces_in_hand == 32: return 0
    elif pieces_in_hand >= 31: return 2   
    elif pieces_in_hand <= 4: return 9
    elif pieces_in_hand <= 8: return 8
    elif pieces_in_hand <= 16: return 7
    elif pieces_in_hand <= 24: return 6
    return 5

def is_timeout(deadline: float):
    if time.perf_counter() >= deadline :
        return True


def max_value(state: State, alpha, beta, cutoff: int, root_player: int, timeout : float):
    if Game.is_terminal(state): return Game.utility(state, root_player), None
    
    if cutoff == 0 or is_timeout(timeout): return evaluation(state, root_player), None

    v = float("-inf")
    best_move = None

    actions = Game.actions(state)

    for a in actions:
        copy_state = state.copy()
        Game.apply(copy_state, a)

        v2, _ = min_value(copy_state, alpha, beta, cutoff - 1, root_player, timeout)

        if v2 > v:
            v = v2
            best_move = a

        alpha = max(alpha, v)
        if alpha >= beta:
            break

    return v, best_move


def min_value(state: State, alpha, beta, cutoff: int, root_player: int, timeout : float):
    if Game.is_terminal(state): return Game.utility(state, root_player), None
    
   
    if cutoff == 0 or is_timeout(timeout):
        return evaluation(state, root_player), None

    v = float("inf")
    best_move = None

    actions = Game.actions(state)

    for a in actions:
        copy_state = state.copy()
        Game.apply(copy_state, a)

        v2, _ = max_value(copy_state, alpha, beta, cutoff - 1, root_player, timeout)

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
        score += line_score(board[i], root_player, opponent, state)
        
        # colonnes
        score += line_score([board[j][i] for j in range(6)], root_player, opponent, state)

    # Petit bonus
    own_pieces = state.pieces_o[root_player] + state.pieces_x[root_player]
    opponent_pieces = state.pieces_o[opponent] + state.pieces_x[opponent]
    score += 0.5 * (own_pieces - opponent_pieces)

    return score


def line_score(cells : list[tuple[str, int]], root_player : int, opponent : int, state : State) :
    score = 0

    for i in range(6) :

        if not cells[i] :
            score += look_forward(cells, root_player, opponent, state, i)
            score += look_backward(cells, root_player, opponent, state, i)

    return score


def look_forward(cells : list[tuple[str, int]], root_player : int, opponent : int, state : State, i : int):
    score = 0
    own2 = adv2 = x_piece2 = o_piece2 = False

    if (i > 0 and not cells[i-1]) or (i <= 2 and not cells[i+3]): empty = 2
    else : empty = 1
    
    if i <= 3 :
        c_plus1, c_plus2 = cells[i+1], cells[i+2]
        if c_plus1 and c_plus2 and empty >= 2:
            if c_plus1[1] == c_plus2[1] == root_player : score += 10 ; own2 = True
            if c_plus1[1] == c_plus2[1] == opponent : score -= 12 ; adv2 = True
            if c_plus1[0] == c_plus2[0] == "x" : x_piece2 = True
            if c_plus1[0] == c_plus2[0] == "o" : o_piece2 = True
            if c_plus1[0] == c_plus2[0] == "x" and state.pieces_x[root_player] >= 2 : score += 10
            if c_plus1[0] == c_plus2[0] == "o" and state.pieces_o[root_player] >= 2 : score += 10
            if c_plus1[0] == c_plus2[0] == "x" and state.pieces_x[opponent] >= 2 : score -= 12
            if c_plus1[0] == c_plus2[0] == "o" and state.pieces_o[opponent] >= 2 : score -= 12

    if i <= 2 and (own2 or adv2 or x_piece2 or o_piece2) :
        c_plus3 = cells[i+3]
        if c_plus3:
            if own2 and c_plus3[1] == root_player : score += 40
            if adv2 and c_plus3[1] == opponent : score -= 50
            if x_piece2 and c_plus3[0] == "x" and state.pieces_x[root_player] >= 1 : score += 40
            if o_piece2 and c_plus3[0] == "o" and state.pieces_o[root_player] >= 1 : score += 40
            if x_piece2 and c_plus3[0] == "x" and state.pieces_x[opponent] >= 1 : score -= 50
            if o_piece2 and c_plus3[0] == "o" and state.pieces_o[opponent] >= 1 : score -= 50

    return score


def look_backward(cells : list[tuple[str, int]], root_player : int, opponent : int, state : State, i : int) :
    score = 0
    own2 = adv2 = x_piece2 = o_piece2 = False

    if (i < 5 and not cells[i+1]) or (i >= 3 and not cells[i-3]) : empty = 2
    else : empty = 1
    
    if i >= 2 :
        c_moin1, c_moin2 = cells[i-1], cells[i-2]
        
        if c_moin2 and c_moin1 and empty >= 2 :
            if c_moin1[1] == c_moin2[1] == root_player : score += 10
            if c_moin1[1] == c_moin2[1] == opponent : score -= 12
            if c_moin1[0] == c_moin2[0] == "x" and state.pieces_x[root_player] >= 2 : score += 10
            if c_moin1[0] == c_moin2[0] == "o" and state.pieces_o[root_player] >= 2 : score += 10
            if c_moin1[0] == c_moin2[0] == "x" and state.pieces_x[opponent] >= 2 : score -= 12
            if c_moin1[0] == c_moin2[0] == "o" and state.pieces_o[opponent] >= 2 : score -= 12
            
    if i >= 3 and (own2 or adv2 or x_piece2 or o_piece2):
        c_moin3 = cells[i-3]
        
        if c_moin3 :
            if i>=4 and c_moin3[1] ==  root_player : score += 40
            if i>=4 and c_moin3[1] == opponent : score -= 50
            if i>=4 and c_moin3[0] == "x" and state.pieces_x[root_player] >= 1 : score += 40
            if i>=4 and c_moin3[0] == "o" and state.pieces_o[root_player] >= 1 : score += 40
            if i>=4 and c_moin3[0] == "x" and state.pieces_x[opponent] >= 1 : score -= 50
            if i>=4 and c_moin3[0] == "o" and state.pieces_o[opponent] >= 1 : score -= 50

    return score