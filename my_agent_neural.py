from agent import Agent
from oxono import Game, State
import time
import random
import torch
import torch.nn as nn
import numpy as np
from oxono_net import OxonoNet, encode_state


class MyAgent(Agent):
    def __init__(self, player):
        super().__init__(player)
        self.cache = {}

    def act(self, state: State, remaining_time):
        actions = Game.actions(state)
        if not actions:
            return None

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


def order_actions(state: State, actions : list[tuple], root_player: int, maximizing: bool):
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


def get_key(state : State, root_player : int):
    board = tuple(tuple(row) for row in state.board)
    totem_x = state.pieces_x[0], state.pieces_x[1]
    totem_o = state.pieces_o[0], state.pieces_o[1]
    return board, state.totem_X, state.totem_O, state.current_player, totem_x, totem_o, root_player
    

def max_value(state: State, alpha : float, beta : float, cutoff: int, root_player: int, deadline: float, cache : dict):
    check_deadline(deadline)

    if Game.is_terminal(state) : return Game.utility(state, root_player), None
    
    key = get_key(state, root_player)

    if key in cache:
        entry = cache[key]
        if entry["depth"] >= cutoff:
            flag, value, best_move = entry["flag"], entry["value"], entry["best_move"]

            if flag == "EXACT" : return value, best_move

            elif flag == "LOWER" : alpha = max(alpha, value)

            elif flag == "UPPER" : beta = min(beta, value)

            if alpha >= beta : return value, best_move
            
    if cutoff == 0 : 
        reseau = OxonoNet()
        reseau.load_state_dict(torch.load("modeles/oxono_net.pth"))
        reseau.eval()
        return evaluation(state, root_player, reseau), None

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

    if v <= alpha_orig:
        flag = "UPPER"
    elif v >= beta_orig:
        flag = "LOWER"
    else:
        flag = "EXACT"

    cache[key] = {"value": v, "best_move": best_move, "depth": cutoff, "flag": flag}

    return v, best_move


def min_value(state: State, alpha : float, beta : float, cutoff: int, root_player: int, deadline: float, cache : dict):
    check_deadline(deadline)

    if Game.is_terminal(state) :
        return Game.utility(state, root_player), None
    
    key = get_key(state, root_player)

    if key in cache:
        entry = cache[key]
        if entry["depth"] >= cutoff:
            flag, value, best_move = entry["flag"], entry["value"], entry["best_move"]

            if flag == "EXACT" : return value, best_move

            elif flag == "LOWER" : alpha = max(alpha, value)

            elif flag == "UPPER" : beta = min(beta, value)

            if alpha >= beta : return value, best_move

    if cutoff == 0 : 
        reseau = OxonoNet()
        reseau.load_state_dict(torch.load("modeles/oxono_net.pth"))
        reseau.eval()
        return evaluation(state, root_player, reseau), None

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

    if v <= alpha_orig:
        flag = "UPPER"
    elif v >= beta_orig:
        flag = "LOWER"
    else:
        flag = "EXACT"

    cache[key] = {"value": v, "best_move": best_move, "depth": cutoff, "flag": flag}    

    return v, best_move


def evaluation(state : State, root_player : int, reseau : OxonoNet):
    if Game.is_terminal(state):
        return Game.utility(state, root_player)

    tenseur = encode_state(state.board, root_player)
    with torch.no_grad():
        score = reseau(tenseur)
    return score.item()