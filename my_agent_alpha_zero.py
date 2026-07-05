from agent import Agent
from oxono import Game, State
import time
import random

"""
╔══════════════════════════════════════════════════════════╗
║              RÉSEAU DE NEURONES  f_θ(s)                  ║
║         entrée : état s   sorties : p(a|s), v(s)         ║
╚══════════════════════════════════════════════════════════╝

"""

def neural_network(state : State) :
    

fonction reseau_neuronal f_θ(s):

    # ── Couche d'entrée ─────────────────────────────────
    x ← encoder(s)
    # Transforme le plateau en tenseur numérique.
    # Ex pour oxono 6×6 : plusieurs plans binaires empilés
    #   plan 0 : 1 là où root_player a une pièce, 0 sinon
    #   plan 1 : 1 là où opponent a une pièce, 0 sinon
    #   plan 2 : 1 partout si c'est le tour de root_player
    #   → tenseur de forme (6, 6, nb_plans)

    # ── Couches résiduelles (corps du réseau) ───────────
    x ← conv(x, filtres=256, taille=3×3)
    x ← batch_normalisation(x)
    x ← relu(x)
    # Conv extrait des patterns locaux (alignements, menaces).
    # Batch norm stabilise l'entraînement.
    # ReLU introduit la non-linéarité (sans elle le réseau
    # ne serait qu'une simple transformation linéaire).

    répéter N fois:   # N = 19 pour AlphaGo Zero
        résidu ← x
        x ← conv(x, filtres=256, taille=3×3)
        x ← batch_normalisation(x)
        x ← relu(x)
        x ← conv(x, filtres=256, taille=3×3)
        x ← batch_normalisation(x)
        x ← x + résidu   # ← connexion résiduelle (skip)
        x ← relu(x)
        # La connexion résiduelle additionne l'entrée du bloc
        # à sa sortie. Cela évite le problème de "gradient
        # qui disparaît" dans les réseaux très profonds :
        # le gradient peut toujours circuler via le chemin court.

    # ── Tête Policy ─────────────────────────────────────
    p ← conv(x, filtres=2, taille=1×1)
    p ← batch_normalisation(p)
    p ← relu(p)
    p ← flatten(p)
    p ← dense(p, taille=nb_actions)
    p ← softmax(p)
    # Softmax convertit des scores bruts en probabilités
    # qui somment à 1. p[a] = probabilité de jouer l'action a.
    # C'est le "prior" : ce que le réseau croit être bon
    # AVANT toute simulation MCTS.

    # ── Tête Value ───────────────────────────────────────
    v ← conv(x, filtres=1, taille=1×1)
    v ← batch_normalisation(v)
    v ← relu(v)
    v ← flatten(v)
    v ← dense(v, taille=256)
    v ← relu(v)
    v ← dense(v, taille=1)
    v ← tanh(v)
    # tanh borne la sortie dans [-1, +1].
    # v ≈ +1 : le réseau pense que root_player va gagner.
    # v ≈ -1 : le réseau pense qu'il va perdre.
    # v remplace le rollout aléatoire du MCTS classique.

    retourner p, v


╔══════════════════════════════════════════════════════════╗
║           MCTS GUIDÉ PAR LE RÉSEAU                       ║
╚══════════════════════════════════════════════════════════╝

fonction MCTS_AlphaZero(état, f_θ, nb_simulations):
    racine ← NœudMCTS(état)

    # Initialisation de la racine par le réseau
    p, v ← f_θ(état)
    pour chaque action a légale:
        racine.prior[a] ← p[a]   # prior issu du réseau

    répéter nb_simulations fois:

        # ── Sélection avec UCT modifié (PUCT) ───────────
        nœud ← racine
        tant que nœud est entièrement développé ET non terminal:
            nœud ← meilleur_enfant_PUCT(nœud)

        # ── Expansion + évaluation par le réseau ────────
        si nœud n'est pas terminal:
            p, v ← f_θ(nœud.état)
            pour chaque action a légale depuis nœud.état:
                nœud.prior[a] ← p[a]
        sinon:
            v ← utilité(nœud.état)   # +1, 0, -1

        # Pas de rollout aléatoire : v sort directement du réseau

        # ── Rétropropagation ────────────────────────────
        tant que nœud n'est pas None:
            nœud.visites += 1
            nœud.valeur  += v
            v ← -v        # alternance des joueurs
            nœud ← nœud.parent

    retourner action la plus visitée depuis racine


fonction meilleur_enfant_PUCT(nœud):
    c ← constante d'exploration

    pour chaque enfant de nœud.enfants:
        Q ← enfant.valeur / enfant.visites   # exploitation
        U ← c × enfant.prior                 # prior réseau
              × √(nœud.visites) / (1 + enfant.visites)
        score ← Q + U

    retourner enfant avec score maximal


╔══════════════════════════════════════════════════════════╗
║           BOUCLE D'ENTRAÎNEMENT                          ║
╚══════════════════════════════════════════════════════════╝

fonction entrainer_AlphaZero(f_θ):
    mémoire ← []

    répéter indéfiniment:

        # 1. Auto-jeu : générer des parties
        pour chaque partie:
            états, politiques, résultat ← jouer_contre_soi(f_θ)
            mémoire.ajouter(états, politiques, résultat)
            # politiques = distributions de visites MCTS,
            # plus fiables que les priors bruts du réseau

        # 2. Entraînement sur les parties jouées
        pour chaque batch de la mémoire:
            p_prédit, v_prédit ← f_θ(états)

            perte_policy ← -Σ politique × log(p_prédit)
            # Entropie croisée : rapproche les priors
            # des vraies distributions de visites MCTS.

            perte_value  ← Σ (résultat - v_prédit)²
            # MSE : rapproche v(s) du vrai résultat de la partie.

            perte_totale ← perte_policy + perte_value
            mettre_à_jour θ via gradient de perte_totale