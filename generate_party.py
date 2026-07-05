import json
import random
import argparse
from oxono import Game, State
from my_agent_v3 import MyAgent as StrongAgent
from my_agent_ici import MyAgent as IntermediateAgent
from random_agent import RandomAgent


def build_agents(matchup: str, strong_mode: str):
    if matchup == "strong_vs_strong":
        return {0: StrongAgent(0, search_mode=strong_mode), 1: StrongAgent(1, search_mode=strong_mode)}
    if matchup == "strong_vs_intermediate":
        return {0: StrongAgent(0, search_mode=strong_mode), 1: IntermediateAgent(1)}
    if matchup == "intermediate_vs_strong":
        return {0: IntermediateAgent(0), 1: StrongAgent(1, search_mode=strong_mode)}
    if matchup == "strong_vs_random":
        return {0: StrongAgent(0, search_mode=strong_mode), 1: RandomAgent(1)}
    return {0: RandomAgent(0), 1: StrongAgent(1, search_mode=strong_mode)}


def profile_to_search_mode(profile: str) -> str:
    # On garde la qualite en mode normal, et on accelere les profils orientés volume/diversite.
    if profile == "quality":
        return "normal"
    if profile == "explore":
        return "very_fast"
    return "fast"


def pick_matchup(profile: str = "balanced") -> str:
    tirage = random.random()
    if profile == "explore":
        # Mix exploration: plus de diversite pour couvrir des positions atypiques.
        if tirage < 0.40:
            return "strong_vs_strong"
        if tirage < 0.80:
            return random.choice(["strong_vs_intermediate", "intermediate_vs_strong"])
        return random.choice(["strong_vs_random", "random_vs_strong"])

    if profile == "quality":
        # Mix qualite: plus de fort/fort pour des labels plus fiables.
        if tirage < 0.80:
            return "strong_vs_strong"
        if tirage < 0.98:
            return random.choice(["strong_vs_intermediate", "intermediate_vs_strong"])
        return random.choice(["strong_vs_random", "random_vs_strong"])

    # Mix equilibre: 60% fort/fort, 30% fort/intermediaire, 10% fort/random.
    if tirage < 0.60:
        return "strong_vs_strong"
    if tirage < 0.90:
        return random.choice(["strong_vs_intermediate", "intermediate_vs_strong"])
    return random.choice(["strong_vs_random", "random_vs_strong"])


def play_a_party(matchup: str, strong_mode: str):
    """
    Joue une partie entre deux agents random.
    Retourne une liste de (état_encodé, résultat_pour_joueur_0)
    """
    state = State()
    historique : list[tuple[State, int]] = []  # liste de (state, joueur_courant)

    agents = build_agents(matchup, strong_mode)
    player = 0
    remaining_time = 300

    while not Game.is_terminal(state):
        action = agents[player].act(state, remaining_time)

        # On sauvegarde l'état AVANT le coup
        historique.append((state.copy(), state.current_player))

        if not action:
            actions = Game.actions(state)
            action = random.choice(actions) if actions else None
            if action is None:
                break

        Game.apply(state, action)
        player = abs(player - 1)

    # Résultat final : utility() du point de vue du joueur 0
    resultat_joueur0 = Game.utility(state, 0)
    # +1 si joueur 0 gagne, -1 s'il perd, 0 si nul

    # Pour chaque état sauvegardé, on calcule le résultat
    # du point de vue du joueur qui jouait à ce moment-là
    donnees = []
    for etat, joueur in historique:
        if joueur == 0:
            resultat = resultat_joueur0
        else:
            resultat = -resultat_joueur0  # inversé pour joueur 1
        donnees.append((etat, joueur, resultat))

    return donnees


def generer_dataset(nb_parties : int = 2000, fichier : str = "dataset.json", profile: str = "balanced", strong_mode: str | None = None):
    dataset = []
    resolved_mode = strong_mode or profile_to_search_mode(profile)
    compteur_matchups = {
        "strong_vs_strong": 0,
        "strong_vs_intermediate": 0,
        "intermediate_vs_strong": 0,
        "strong_vs_random": 0,
        "random_vs_strong": 0,
    }

    for i in range(nb_parties):
        if i % 100 == 0:
            print(f"Partie {i}/{nb_parties}...")

        matchup = pick_matchup(profile=profile)
        compteur_matchups[matchup] += 1
        donnees_partie = play_a_party(matchup, resolved_mode)

        for etat, joueur, resultat in donnees_partie:
            # On sérialise l'état pour le sauvegarder en JSON
            dataset.append({
                "board": str(etat.board),       # sérialisation
                "pieces_x": list(etat.pieces_x),
                "pieces_o": list(etat.pieces_o),
                "joueur": joueur,
                "resultat": resultat            # -1, 0, ou +1
            })

    with open(fichier, "w") as f:
        json.dump(dataset, f)

    print(f"Dataset sauvegardé : {len(dataset)} états")
    print("Mix de parties:", compteur_matchups)
    print(f"Mode recherche agent fort: {resolved_mode}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generation de dataset Oxono")
    parser.add_argument(
        "--preset",
        choices=["quick", "quality", "explore", "long", "custom"],
        default="quick",
        help="quick=2000 parties, quality=5000 parties (mix plus fort), explore=5000 parties (mix plus diversifie), long=20000 parties, custom utilise --nb-parties",
    )
    parser.add_argument(
        "--nb-parties",
        type=int,
        default=None,
        help="Nombre de parties (utilise seulement avec --preset custom)",
    )
    parser.add_argument(
        "--fichier",
        type=str,
        default=None,
        help="Nom du fichier de sortie JSON",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed aleatoire pour generation reproductible",
    )
    parser.add_argument(
        "--strong-mode",
        choices=["auto", "normal", "fast", "very_fast"],
        default="auto",
        help="Mode de recherche pour l'agent fort (auto selon le preset)",
    )
    return parser.parse_args()


def resolve_run_config(args):
    if args.preset == "quick":
        return 2000, args.fichier or "dataset_quick.json", "balanced"
    if args.preset == "quality":
        return 5000, args.fichier or "dataset_quality.json", "quality"
    if args.preset == "explore":
        return 5000, args.fichier or "dataset_explore.json", "explore"
    if args.preset == "long":
        return 20000, args.fichier or "dataset_long.json", "balanced"

    if args.nb_parties is None or args.nb_parties <= 0:
        raise ValueError("Avec --preset custom, --nb-parties doit etre un entier > 0")
    return args.nb_parties, args.fichier or "dataset_custom.json", "balanced"


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    nb_parties, fichier, profile = resolve_run_config(args)
    strong_mode = None if args.strong_mode == "auto" else args.strong_mode
    print(
        f"Generation preset={args.preset} profile={profile} nb_parties={nb_parties} fichier={fichier}"
    )
    generer_dataset(nb_parties=nb_parties, fichier=fichier, profile=profile, strong_mode=strong_mode)