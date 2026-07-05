from agent import Agent
from oxono import Game, State
import time
import random


class MyAgent(Agent):
    def __init__(self, player):
        super().__init__(player)

    def act(self, state: State, remaining_time):
        actions = Game.actions(state)
        if not actions:
            return None

        max_depth = calculer_depth(state)
        if max_depth == 0:
            return random.choice(actions)

        budget = compute_turn_budget(state, remaining_time, self.player)
        deadline = time.perf_counter() + budget

        best_move = random.choice(actions)

        # Keep the best move from the last fully completed depth.
        for depth in range(1, max_depth + 1):
            try:
                _, move = max_value(
                    state=state,
                    alpha=float("-inf"),
                    beta=float("inf"),
                    cutoff=depth,
                    root_player=self.player,
                    deadline=deadline,
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


def compute_turn_budget(state: State, remaining_time: float, player: int) -> float:
    remaining_moves = state.pieces_x[player] + state.pieces_o[player]
    safe_remaining = max(remaining_time - 0.2, 0.05)

    if remaining_moves <= 1:
        return safe_remaining

    nominal = safe_remaining / remaining_moves
    return max(0.05, min(safe_remaining, nominal * 1.1))


def calculer_depth(state: State):
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
        pr, pc = piece_pos
        symbol = "o" if totem == "o" else "x"

        score = -(abs(pr - center) + abs(pc - center))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = pr + dr, pc + dc
            if not (0 <= nr < 6 and 0 <= nc < 6):
                continue
            cell = state.board[nr][nc]
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


def max_value(state: State, alpha, beta, cutoff: int, root_player: int, deadline: float):
    check_deadline(deadline)

    if Game.is_terminal(state):
        return Game.utility(state, root_player), None

    if cutoff == 0:
        return evaluation(state, root_player), None

    v = float("-inf")
    best_move = None

    actions = order_actions(state, Game.actions(state), root_player, maximizing=True)
    for action in actions:
        check_deadline(deadline)

        copy_state = state.copy()
        Game.apply(copy_state, action)

        v2, _ = min_value(copy_state, alpha, beta, cutoff - 1, root_player, deadline)

        if v2 > v:
            v = v2
            best_move = action

        alpha = max(alpha, v)
        if alpha >= beta:
            break

    return v, best_move


def min_value(state: State, alpha, beta, cutoff: int, root_player: int, deadline: float):
    check_deadline(deadline)

    if Game.is_terminal(state):
        return Game.utility(state, root_player), None

    if cutoff == 0:
        return evaluation(state, root_player), None

    v = float("inf")
    best_move = None

    actions = order_actions(state, Game.actions(state), root_player, maximizing=False)
    for action in actions:
        check_deadline(deadline)

        copy_state = state.copy()
        Game.apply(copy_state, action)

        v2, _ = max_value(copy_state, alpha, beta, cutoff - 1, root_player, deadline)

        if v2 < v:
            v = v2
            best_move = action

        beta = min(beta, v)
        if alpha >= beta:
            break

    return v, best_move


def evaluation(state: State, root_player: int):
    if Game.is_terminal(state):
        return Game.utility(state, root_player)

    opponent = 1 - root_player
    board = state.board

    score = 0

    for i in range(6):
        score += line_score(board[i], root_player, opponent, state)
        score += line_score([board[j][i] for j in range(6)], root_player, opponent, state)

    # Bonus léger sur le nombre de pièces
    own_pieces = state.pieces_o[root_player] + state.pieces_x[root_player]
    opp_pieces = state.pieces_o[opponent] + state.pieces_x[opponent]
    score += 0.3 * (own_pieces - opp_pieces)

    return score


def line_score(cells, root_player, opponent, state):
    score = 0
    
    own = adv = empty = 0
    x_count = o_count = 0

    for c in cells:
        if c is None:
            empty += 1
        else:
            symbol, player = c
            if player == root_player:
                own += 1
            else:
                adv += 1

            if symbol == "x":
                x_count += 1
            else:
                o_count += 1
    
    if own > 0 and adv > 0:
        
        score += symbol_potential(x_count, o_count, state, root_player, opponent)
        return score

    # Agent
    if own == 3 and empty >= 1:
        score += 50
    elif own == 2 and empty >= 2:
        score += 12
    elif own == 1 and empty >= 3:
        score += 3

    if adv == 3 and empty >= 1:
        score -= 60
    elif adv == 2 and empty >= 2:
        score -= 15
    elif adv == 1 and empty >= 3:
        score -= 3
    
    score += symbol_potential(x_count, o_count, state, root_player, opponent)

    return score


def symbol_potential(x_count, o_count, state, root_player, opponent):
    score = 0

    # X
    if x_count == 3:
        if state.pieces_x[root_player] > 0:
            score += 25
        if state.pieces_x[opponent] > 0:
            score -= 25

    # O
    if o_count == 3:
        if state.pieces_o[root_player] > 0:
            score += 25
        if state.pieces_o[opponent] > 0:
            score -= 25

    return score