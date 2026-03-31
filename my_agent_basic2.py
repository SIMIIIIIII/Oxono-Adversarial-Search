from agent import Agent
from oxono import Game, State
import time
import random


class MyAgent(Agent):
    def __init__(self, player):
        super().__init__(player)

    def act(self, state: State, remaining_time):
        remaining_pieces = 16 - state.pieces_x[self.player] - state.pieces_o[self.player]

        timout = max((remaining_time / max(remaining_pieces, 1)) * 0.8, 0.5)
        start = time.time()
        cutoff = calculer_depth(state)

        if cutoff == 0 : return random.choice(Game.actions(state))

        value, move = max_value(
            state=state,
            alpha=float("-inf"),
            beta=float("inf"),
            cutoff=cutoff,
            root_player=self.player,
            timout=timout,
            start=start
        )
        
        if move is None:
            actions = Game.actions(state)
            return actions[0] if actions else None
        return move
    

def calculer_depth(state : State):
    nb_pieces = 32 - state.pieces_x[0] - state.pieces_x[1] - state.pieces_o[0] - state.pieces_o[1]

    if nb_pieces == 0 : return 0

    if nb_pieces < 10 : return 5
    elif nb_pieces < 24 : return 6
    else : return 6 + (nb_pieces - 24) // 3


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
        score += line_score(board[i], root_player, opponent, state)
        
        # colonnes
        score += line_score([board[j][i] for j in range(6)], root_player, opponent, state)

    # Petit bonus
    own_pieces = state.pieces_o[root_player] + state.pieces_x[root_player]
    opponent_pieces = state.pieces_o[opponent] + state.pieces_x[opponent]
    score += 0.5 * (own_pieces - opponent_pieces)

    return score


def line_score(cells : list[tuple[str, int]], root_player : int, opponent : int, state : State) :
    own2 = own1 = False
    adv2 = adv1 = False
    x_piece2 = x_piece1 = False
    o_piece2 = o_piece1 = False
    empty = score = 0

    for i in range(6) :

        if not cells[i] :
            if i <= 3 :
                c_plus1, c_plus2 = cells[i+1], cells[i+2]
                if c_plus1 and c_plus2 :
                    if [c_plus1[1],  c_plus2[1]].count(root_player) == 2 : score += 10 ; own2 = True
                    if [c_plus1[1],  c_plus2[1]].count(opponent) == 2 : score -= 12 ; adv2 = True
                    if [c_plus1[0],  c_plus2[0]].count("x") == 2 : x_piece2 = True
                    if [c_plus1[0],  c_plus2[0]].count("o") == 2 : o_piece2 = True
                    if [c_plus1[0],  c_plus2[0]].count("x") == 2 and state.pieces_x[root_player] >= 2 : score += 10
                    if [c_plus1[0],  c_plus2[0]].count("o") == 2 and state.pieces_o[root_player] >= 2 : score += 10
                    if [c_plus1[0],  c_plus2[0]].count("x") == 2 and state.pieces_x[opponent] >= 2 : score -= 12
                    if [c_plus1[0],  c_plus2[0]].count("o") == 2 and state.pieces_o[opponent] >= 2 : score -= 12

            if i <= 2 and (own2 or adv2 or x_piece2 or o_piece2) :
                c_plus3 = cells[i+3]

                if c_plus3:
                    if own2 and c_plus3[1] == root_player : score += 40
                    if adv2 and c_plus3[1] == opponent : score -= 50
                    if x_piece2 and c_plus3[0] == "x" and state.pieces_x[root_player] >= 1 : score += 40
                    if o_piece2 and c_plus3[0] == "o" and state.pieces_o[root_player] >= 1 : score += 40
                    if x_piece2 and c_plus3[0] == "x" and state.pieces_x[opponent] >= 1 : score -= 50
                    if o_piece2 and c_plus3[0] == "o" and state.pieces_o[opponent] >= 1 : score -= 50

            if i >= 2 :
                c_moin1, c_moin2 = cells[i-1], cells[i-2]

                if c_moin2 and c_moin1 :
                    if i>=4 and [c_moin1[1],  c_moin2[1]].count(root_player) == 2 : score += 10
                    if i>=4 and [c_moin1[1],  c_moin2[1]].count(opponent) == 2: score -= 12
                    if i>=4 and [c_moin1[0],  c_moin2[0]].count("x") == 2 and state.pieces_x[root_player] >= 2 : score += 10
                    if i>=4 and [c_moin1[0],  c_moin2[0]].count("o") == 2 and state.pieces_o[root_player] >= 2 : score += 10
                    if i>=4 and [c_moin1[0],  c_moin2[0]].count("x") == 2 and state.pieces_x[opponent] >= 2 : score -= 12
                    if i>=4 and [c_moin1[0],  c_moin2[0]].count("o") == 2 and state.pieces_o[opponent] >= 2 : score -= 12

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