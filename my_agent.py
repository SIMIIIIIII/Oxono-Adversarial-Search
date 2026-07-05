from agent import Agent
from oxono import Game, State
import time
import random
import math


class MyAgent(Agent):
    def __init__(self, player):
        super().__init__(player)
        self.exploration = math.sqrt(2.0)

    def act(self, state: State, remaining_time):
        actions = Game.actions(state)
        if not actions:
            return None

        # S'il y a un mouvement gagnant immédiat, il le joue
        winning_move = find_winning_action(state)
        if winning_move is not None:
            return winning_move

        budget = get_budget(state, remaining_time, self.player)
        timeout= time.perf_counter() + budget

        try:
            return mcts(state, timeout, self.player, self.exploration)
        except SearchTimeout:
            return random.choice(actions)
    
class SearchTimeout(Exception):
    pass


def check_deadline(deadline: float):
    """ Leve une exception si le temps imparti est écoulé.
    """
    if time.perf_counter() >= deadline:
        raise SearchTimeout()


def get_budget(state: State, remaining_time: float, player: int) -> float:
    """Retourne le timeout pour ce tours
    """
    remaining_moves = state.pieces_x[player] + state.pieces_o[player]
    safe_remaining = max(remaining_time - 0.2, 0.05) # Minimum 0.05 seconde de recherche

    if remaining_moves <= 1:
        return safe_remaining

    # Le bugdet c'est le temps diviser par les nombres d'actions restantes
    budget = safe_remaining / remaining_moves 
    
    return max(0.05, min(safe_remaining, 6.0, budget * 1.1))


class Node:
    def __init__(self, state: State, root_player: int, parent=None, action_from_parent=None):
        self.state : State = state
        self.root_player : int = root_player
        self.parent = parent # Le node parent d'où il a été généré
        self.action_from_parent = action_from_parent # L'action qui a générée ce node

        self.children : list[Node] = [] # La liste de toutes les actions possibles à partir de ce node
        self.untried_actions = Game.actions(state) # Les actions non encore essayées
        self.visits = 0 # Le nombre de fois ou ce node a été visité
        self.value_sum = 0.0

    def is_terminal(self) -> bool:
        """ Retourne si le node contient l'état finale
        """
        return Game.is_terminal(self.state)

    def is_fully_expanded(self) -> bool:
        """ Retourne si toutes les actions de ce node ont déjà été essayée
        """
        return len(self.untried_actions) == 0

    def expand_one(self):
        """ Genere et return un node grace à une action dans du node actuel
        """
        action = pick_action(self.state, self.untried_actions)
        next_state = self.state.copy()
        Game.apply(next_state, action)

        child = Node(next_state, self.root_player, parent=self, action_from_parent=action)
        self.children.append(child)
        return child

    def best_child_uct(self, exploration: float):
        """ Retourne le noeud avec le meilleur score utc
        UTC = (Qc/Nc) + C * (sqrt(len(Np) / Nc))
        """
        best_score = float("-inf")
        best_child = None
        log_parent = math.log(self.visits)

        for child in self.children:
            if child.visits == 0:
                return child

            exploit = child.value_sum / child.visits

            # Inverse le poid en negatif quand c'est le tours de l'adversaire
            if Game.to_move(self.state) != self.root_player:
                exploit = -exploit

            explore = exploration * math.sqrt(log_parent / child.visits)
            score = exploit + explore

            if score > best_score:
                best_score = score
                best_child = child

        return best_child


def find_winning_action(state: State) -> tuple | None:
    """ l’agent vérifie si un coup gagnant direct existe
    """
    player : int = Game.to_move(state)
    actions = Game.actions(state)

    for action in actions:
        next_state = state.copy()
        Game.apply(next_state, action)

        if Game.is_terminal(next_state) and Game.utility(next_state, player) == 1:
            return action

    return None


def pick_action(state: State, untried_actions: list):
    """ Retourne une action non testée menant à une victoire immédiate en prioritée.
        Sinon, retourne une action ramdom
    """
    player : int = Game.to_move(state)

    for i, action in enumerate(untried_actions):
        next_state = state.copy()
        Game.apply(next_state, action)
        if Game.is_terminal(next_state) and Game.utility(next_state, player) == 1:
            return untried_actions.pop(i)

    return untried_actions.pop(random.randrange(len(untried_actions)))


def play(state: State, root_player: int, deadline: float) -> int:
    """ Joue une action gagnante d'abord sinon une action au hasard
    """
    copy_state : State = state.copy()

    while not Game.is_terminal(copy_state):
        check_deadline(deadline)
        action = find_winning_action(copy_state)

        if action is None:
            actions = Game.actions(copy_state)
            if not actions:
                break
            action = random.choice(actions)
        Game.apply(copy_state, action)

    return Game.utility(copy_state, root_player)


def select_and_expand(root: Node, deadline: float, exploration: float) -> Node:
    """ Selectionne un node final ou génère un node à partir de ce node
    """
    node : Node = root

    while True:
        check_deadline(deadline)

        if node.is_terminal():
            return node

        if not node.is_fully_expanded():
            return node.expand_one()

        node = node.best_child_uct(exploration)


def backpropagate(node: Node, result: float) -> None:
    """ Fait remonter le score ou poid vers le node parent
    """
    current = node
    while current is not None:
        current.visits += 1
        current.value_sum += result
        current = current.parent


def best_move_from_root(root: Node):
    """ Retourne le meilleur action depuis la racine s'il y a des actions disponible
        Sinon None
    """
    if not root.children:
        return None

    # prend l’enfant le plus exploré, si egalité rendre celui qui a le meilleur rendement moyen
    best_child = max(
        root.children,
        key=lambda c: (c.visits, (c.value_sum / c.visits) if c.visits else float("-inf"))
    )
    return best_child.action_from_parent


def mcts(state: State, deadline: float, root_player: int, exploration: float):
    root = Node(state.copy(), root_player)

    if not root.untried_actions:
        return None

    try:
        while True:
            leaf = select_and_expand(root, deadline, exploration)
            result = play(leaf.state, root_player, deadline)
            backpropagate(leaf, result)
    except SearchTimeout:
        move = best_move_from_root(root)
        if move is None:
            actions = Game.actions(state)
            return random.choice(actions) if actions else None
        return move