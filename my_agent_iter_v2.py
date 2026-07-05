from agent import Agent
from oxono import Game, State
import time
import random


TT_MAX_ENTRIES = 60000
TT_MIN_CUTOFF = 2


class MyAgent(Agent):
    def __init__(self, player):
        super().__init__(player)
        self.cache = {}

    def act(self, state: State, remaining_time):
        actions = Game.actions(state)
        if not actions:
            return None

        # Keep TT local to one move: useful for iterative deepening while capping growth.
        self.cache.clear()

        max_depth = get_depth(state)
        if max_depth == 0:
            return random.choice(actions)

        budget = get_budget(state, remaining_time, self.player)
        deadline = time.perf_counter() + budget

        best_move = random.choice(actions)

        # Iterative deepening.
        for depth in range(1, max_depth + 1):
            try:
                _, move = max_value(
                    state=state,
                    alpha=float("-inf"),
                    beta=float("inf"),
                    cutoff=depth,
                    root_player=self.player,
                    deadline=deadline,
                    cache=self.cache
                )
                if move is not None:
                    best_move = move
            except SearchTimeout:
                break

        return best_move
    
class SearchTimeout(Exception):
    pass


def check_deadline(deadline: float):
    if time.perf_counter() >= deadline:
        raise SearchTimeout()


def get_budget(state: State, remaining_time: float, player: int) -> float:
    remaining_moves = state.pieces_x[player] + state.pieces_o[player]
    safe_remaining = max(remaining_time - 0.2, 0.05)

    if remaining_moves <= 1:
        return safe_remaining

    nominal = safe_remaining / remaining_moves
    return max(0.05, min(safe_remaining, nominal * 1.1))
    

def get_depth(state : State):
    nb_pieces = 32 - state.pieces_x[0] - state.pieces_x[1] - state.pieces_o[0] - state.pieces_o[1]
    
    if nb_pieces <= 2 : return nb_pieces
    if nb_pieces < 10 : return 5
    if nb_pieces < 24 : return 6
    return 6 + (nb_pieces - 24) // 3


def order_actions(state: State, actions, root_player: int, maximizing: bool):
    player = Game.to_move(state)
    center = 2.5

    def action_priority(action):
        totem, _, piece_pos = action
        piece_row, piece_col = piece_pos
        symbol = "o" if totem == "o" else "x"

        score = -(abs(piece_row - center) + abs(piece_col - center))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor_row, neighbor_col = piece_row + dr, piece_col + dc
            if not (0 <= neighbor_row < 6 and 0 <= neighbor_col < 6):
                continue
            cell = state.board[neighbor_row][neighbor_col]
            if cell is None:
                continue
            if cell[1] == player:
                score += 2.5
            if cell[0] == symbol:
                score += 2.0
            if cell[1] != player:
                score += 0.8

        if player == root_player:
            score += 0.1

        return score

    return sorted(actions, key=action_priority, reverse=maximizing)


def get_key(state : State, root_player):
    board = state.board
    totem_x = state.pieces_x[0], state.pieces_x[1]
    totem_o = state.pieces_o[0], state.pieces_o[1]
    return (
        tuple(board[0]), tuple(board[1]), tuple(board[2]),
        tuple(board[3]), tuple(board[4]), tuple(board[5]),
        state.totem_X,
        state.totem_O,
        state.current_player,
        totem_x,
        totem_o,
        root_player,
    )


def tt_probe(cache: dict, key, cutoff: int, alpha, beta):
    if cutoff < TT_MIN_CUTOFF:
        return alpha, beta, None, None, False

    entry = cache.get(key)
    if entry is None or entry["depth"] < cutoff:
        return alpha, beta, None, None, False

    flag, value, best_move = entry["flag"], entry["value"], entry["best_move"]

    if flag == "EXACT":
        return alpha, beta, value, best_move, True

    if flag == "LOWER":
        alpha = max(alpha, value)
    elif flag == "UPPER":
        beta = min(beta, value)

    if alpha >= beta:
        return alpha, beta, value, best_move, True

    return alpha, beta, None, None, False


def tt_store(cache: dict, key, cutoff: int, value, best_move, alpha_orig, beta_orig):
    if cutoff < TT_MIN_CUTOFF:
        return

    if value <= alpha_orig:
        flag = "UPPER"
    elif value >= beta_orig:
        flag = "LOWER"
    else:
        flag = "EXACT"

    if len(cache) >= TT_MAX_ENTRIES:
        cache.clear()

    cache[key] = {"value": value, "best_move": best_move, "depth": cutoff, "flag": flag}
    

def max_value(state: State, alpha, beta, cutoff: int, root_player: int, deadline: float, cache : dict):
    check_deadline(deadline)

    if Game.is_terminal(state) : return Game.utility(state, root_player), None

    if cutoff == 0: return evaluation(state, root_player), None

    key = get_key(state, root_player)
    alpha, beta, tt_value, tt_move, done = tt_probe(cache, key, cutoff, alpha, beta)
    if done:
        return tt_value, tt_move

    alpha_orig = alpha
    beta_orig = beta

    v = float("-inf")
    best_move = None

    actions = order_actions(state, Game.actions(state), root_player, maximizing=True)
    for a in actions:
        check_deadline(deadline)

        copy_state = state.copy()
        Game.apply(copy_state, a)

        v2, _ = min_value(copy_state, alpha, beta, cutoff - 1, root_player, deadline, cache)

        if v2 > v:
            v = v2
            best_move = a

        alpha = max(alpha, v)
        if alpha >= beta:
            break

    tt_store(cache, key, cutoff, v, best_move, alpha_orig, beta_orig)

    return v, best_move


def min_value(state: State, alpha, beta, cutoff: int, root_player: int, deadline: float, cache : dict):
    check_deadline(deadline)

    if Game.is_terminal(state) :
        return Game.utility(state, root_player), None

    if cutoff == 0:
        return evaluation(state, root_player), None

    key = get_key(state, root_player)
    alpha, beta, tt_value, tt_move, done = tt_probe(cache, key, cutoff, alpha, beta)
    if done:
        return tt_value, tt_move

    alpha_orig = alpha
    beta_orig = beta

    v = float("inf")
    best_move = None

    actions = order_actions(state, Game.actions(state), root_player, maximizing=False)
    for a in actions:
        check_deadline(deadline)

        copy_state = state.copy()
        Game.apply(copy_state, a)

        v2, _ = max_value(copy_state, alpha, beta, cutoff - 1, root_player, deadline, cache)

        if v2 < v:
            v = v2
            best_move = a

        beta = min(beta, v)
        if alpha >= beta:
            break

    tt_store(cache, key, cutoff, v, best_move, alpha_orig, beta_orig)

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
            empty += 0
            if i <= 3 :
                c_plus1, c_plus2 = cells[i+1], cells[i+2]
                if c_plus1 and c_plus2 and empty >= 2:
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