from agent import Agent
from oxono import Game, State
import math
import random
import time


class SearchTimeout(Exception):
    pass


class Node:
    def __init__(self, state: State, root_player: int, parent=None, action_from_parent=None):
        self.state = state
        self.root_player = root_player
        self.parent = parent
        self.action_from_parent = action_from_parent

        self.children = []
        self.untried_actions = Game.actions(state)
        self.visits = 0
        self.value_sum = 0.0

    def is_terminal(self):
        return Game.is_terminal(self.state)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def expand_one(self):
        action = self.untried_actions.pop(random.randrange(len(self.untried_actions)))
        next_state = self.state.copy()
        Game.apply(next_state, action)
        child = Node(next_state, self.root_player, parent=self, action_from_parent=action)
        self.children.append(child)
        return child

    def best_child_uct(self, exploration: float):
        best_score = float("-inf")
        best_child = None
        log_parent = math.log(self.visits)

        for child in self.children:
            if child.visits == 0:
                return child

            exploit = child.value_sum / child.visits
            explore = exploration * math.sqrt(log_parent / child.visits)
            score = exploit + explore

            if score > best_score:
                best_score = score
                best_child = child

        return best_child


def check_deadline(deadline: float):
    if time.perf_counter() >= deadline:
        raise SearchTimeout()


def get_budget(state: State, remaining_time: float, player: int) -> float:
    remaining_moves = state.pieces_x[player] + state.pieces_o[player]
    safe_remaining = max(remaining_time - 0.2, 0.05)

    if remaining_moves <= 1:
        return safe_remaining

    nominal = safe_remaining / remaining_moves
    return max(0.05, min(safe_remaining, 2.0, nominal))


def select_and_expand(root: Node, deadline: float, exploration: float):
    node = root

    while True:
        check_deadline(deadline)

        if node.is_terminal():
            return node

        if not node.is_fully_expanded():
            return node.expand_one()

        node = node.best_child_uct(exploration)


def random_playout(state: State, root_player: int, deadline: float):
    sim_state = state.copy()

    while not Game.is_terminal(sim_state):
        check_deadline(deadline)
        actions = Game.actions(sim_state)
        if not actions:
            break
        action = random.choice(actions)
        Game.apply(sim_state, action)

    return Game.utility(sim_state, root_player)


def backpropagate(node: Node, result: float):
    cur = node
    while cur is not None:
        cur.visits += 1
        cur.value_sum += result
        cur = cur.parent


def best_move_from_root(root: Node):
    if not root.children:
        return None
    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.action_from_parent


def mcts(state: State, deadline: float, root_player: int, exploration: float):
    root = Node(state.copy(), root_player)

    if not root.untried_actions:
        return None

    try:
        while True:
            leaf = select_and_expand(root, deadline, exploration)
            result = random_playout(leaf.state, root_player, deadline)
            backpropagate(leaf, result)
    except SearchTimeout:
        move = best_move_from_root(root)
        if move is None:
            actions = Game.actions(state)
            return random.choice(actions) if actions else None
        return move


class MyAgent(Agent):
    def __init__(self, player):
        super().__init__(player)
        self.exploration = math.sqrt(2.0)

    def act(self, state: State, remaining_time: float):
        actions = Game.actions(state)
        if not actions:
            return None

        budget = get_budget(state, remaining_time, self.player)
        deadline = time.perf_counter() + budget

        try:
            return mcts(state, deadline, self.player, self.exploration)
        except SearchTimeout:
            return random.choice(actions)
