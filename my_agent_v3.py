from agent import Agent
from oxono import Game, State
import time
import random


class MyAgent(Agent):
    def __init__(self, player, search_mode: str = "normal"):
        super().__init__(player)
        self.search_mode = search_mode

    def act(self, state: State, remaining_time):

        timeout = time.perf_counter() + get_budget(state, remaining_time, self.player, self.search_mode)
        cutoff = get_depth(state, self.search_mode)

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
    

def get_budget(state: State, remaining_time: float, player: int, search_mode: str = "normal") -> float:
    remaining_moves = state.pieces_x[player] + state.pieces_o[player]
    safe_remaining = max(remaining_time - 0.2, 0.05)

    if remaining_moves <= 1:
        return safe_remaining

    nominal = safe_remaining / remaining_moves
    # Mode generation: on reduit le budget sans casser la logique adaptative.
    budget_scale = {
        "normal": 5.0,
        "fast": 3.0,
        "very_fast": 2.0,
    }.get(search_mode, 5.0)

    return max(0.05, min(safe_remaining, nominal * budget_scale))
    

def get_depth(state: State, search_mode: str = "normal") -> int:
    # Nombre total de pièces encore en main (les deux joueurs)
    pieces_in_hand = (
        state.pieces_x[0] + state.pieces_x[1] +
        state.pieces_o[0] + state.pieces_o[1]
    )
    
    # Plus il y a de pièces en main → plus d'actions → moins de profondeur
    if pieces_in_hand == 32:
        base_depth = 0
    elif pieces_in_hand >= 31:
        base_depth = 2   # Fin de partie : que des mouvements, branching très faible
    elif pieces_in_hand <= 4:
        base_depth = 9
    elif pieces_in_hand <= 8:
        base_depth = 8
    elif pieces_in_hand <= 16:
        base_depth = 7
    elif pieces_in_hand <= 24:
        base_depth = 6
    else:
        base_depth = 5        # Début de partie : beaucoup de placements possibles

    depth_delta = {
        "normal": 0,
        "fast": -1,
        "very_fast": -2,
    }.get(search_mode, 0)

    # On garde au moins 1 ply de recherche (sauf cas special ouverture a 0).
    if base_depth == 0:
        return 0
    return max(1, base_depth + depth_delta)

def is_timeout(deadline: float):
    if time.perf_counter() >= deadline :
        print("timeout")
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


def line_score(cells : list, root_player : int, opponent : int, state : State) :
    score = 0
    own_total = state.pieces_o[root_player] + state.pieces_x[root_player]
    opp_total = state.pieces_o[opponent] + state.pieces_x[opponent]

    # j -> (bonus attaque, malus défense)
    weights = {
        1: (2, 2),
        2: (10, 12),
        3: (40, 50),
    }

    for i in range(3):
        window = cells[i:i + 4]

        own = adv = empty = 0
        x_count = o_count = 0

        for c in window:
            if c is None:
                empty += 1
            else:
                symbol, player = c
                if player == root_player:
                    own += 1
                elif player == opponent:
                    adv += 1

                if symbol == "x":
                    x_count += 1
                elif symbol == "o":
                    o_count += 1

        # 1) Canal joueur: signal principal, stable.
        # Si la fenêtre contient les deux joueurs, on neutralise ce canal.
        if not (own > 0 and adv > 0):
            for j, (add, moin) in weights.items():
                n = 4 - j
                if empty == n:
                    if own == j and own_total >= n:
                        score += add
                    if adv == j and opp_total >= n:
                        score -= moin

        # 2) Canal symbole: utilisé surtout quand le canal joueur est mixte.
        # Cela évite de surcompter la même menace deux fois.
        if own > 0 and adv > 0:
            for j, (add, moin) in weights.items():
                n = 4 - j
                if empty == n:
                    if x_count == j:
                        if state.pieces_x[root_player] >= n:
                            score += add
                        if state.pieces_x[opponent] >= n:
                            score -= moin
                    if o_count == j:
                        if state.pieces_o[root_player] >= n:
                            score += add
                        if state.pieces_o[opponent] >= n:
                            score -= moin

    return score