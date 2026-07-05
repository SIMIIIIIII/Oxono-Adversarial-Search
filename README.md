# Oxono-Adversarial-Search

Projet d'IA adversariale autour du jeu Oxono.

Objectif: concevoir, comparer et analyser plusieurs agents (random, heuristiques, minimax/alpha-beta, MCTS, variantes itératives, version neuronale), avec gestion stricte du temps et infrastructure de benchmark reproductible.

## Points clés

- Moteur de jeu complet (règles, génération d'actions, utilité terminale).
- Exécution isolée des agents dans des sous-processus pour robustesse.
- Contrôle de timeout par joueur (horloge globale par partie).
- Batch de matchs + logs + script d'analyse agrégée.
- Interface visuelle pour jouer contre un agent ou observer un match.
- Pipeline optionnel de self-play et entraînement de réseau de valeur.

## Structure du projet

- `oxono.py`: logique du jeu (état, actions, transitions, terminal, utilité).
- `agent.py`: interface de base à implémenter (`act(state, remaining_time)`).
- `manager.py`: exécute N parties entre 2 agents, imprime les stats, génère des logs.
- `visual_manager.py`: interface Pygame pour match interactif (humain/agent).
- `replayer.py`: relecture d'un log de partie.
- `analyze_results.py`: synthèse des résultats de plusieurs matchups.
- `random_agent.py`: baseline aléatoire.
- `my_agent*.py`: différentes versions d'agents expérimentaux.
- `generate_party.py`, `training.py`, `oxono_net.py`: génération dataset + entraînement réseau (optionnel).

## Prérequis

- Python 3.10+
- Linux, macOS ou Windows

Installation minimale:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dépendances additionnelles pour la partie neuronale:

```bash
pip install torch numpy
```

## Démarrage rapide

### 1) Lancer un match en terminal

```bash
python3 manager.py -n 1 -p0 my_agent.py -p1 random_agent.py
```

### 2) Lancer un benchmark multi-parties avec logs

```bash
python3 manager.py -n 50 -p0 my_agent.py -p1 my_agent_basic.py -l logs_ablation -t 300 > result_ablation.txt
```

### 3) Analyser des fichiers de résultats

```bash
python3 analyze_results.py result_ablation.txt
```

Ou sans argument (analyse un set de fichiers par défaut si présents):

```bash
python3 analyze_results.py
```

## Interface visuelle

Humain (rose) vs agent (noir):

```bash
python3 visual_manager.py -p0 human -p1 my_agent.py -t 300
```

Deux humains:

```bash
python3 visual_manager.py -p0 human -p1 human
```

Contrôles:

- Clic pour sélectionner le totem, sa destination, puis la case de pose.
- Échap pour fermer.

## Rejouer une partie enregistrée

```bash
python3 replayer.py logs_ablation/log_0.txt
```

Contrôles:

- Flèche droite: coup suivant
- Flèche gauche: coup précédent
- Échap: quitter

## Créer son propre agent

1. Créer un fichier, par exemple `my_new_agent.py`.
2. Définir une classe qui hérite de `Agent`.
3. Implémenter `act(state, remaining_time)` et retourner une action légale de `Game.actions(state)`.
4. Tester:

```bash
python3 manager.py -n 20 -p0 my_new_agent.py -p1 random_agent.py
```

## Pipeline optionnel: dataset + entraînement

Génération de dataset par self-play:

```bash
python3 generate_party.py --preset quick --seed 42
```

Entraînement du réseau:

```bash
python3 training.py --dataset dataset_quick.json --epochs 10 --batch-size 64 --lr 0.001
```

Les poids sont enregistrés dans `modeles/`.

## Reproductibilité

- Utiliser un `--seed` fixe sur les scripts qui l'exposent.
- Garder les mêmes paramètres `-t`, `-n`, `-p0`, `-p1` lors des comparaisons.
- Jouer les matchups aller/retour pour réduire l'effet du premier joueur.

## Limites actuelles

- Le niveau et la vitesse varient selon la version d'agent sélectionnée.
- La partie neuronale nécessite des dépendances supplémentaires non incluses dans `requirements.txt`.
- Les performances peuvent dépendre fortement de la machine (CPU, disponibilité mémoire).

## Auteur

Projet réalisé dans le cadre du cours d'Intelligence Artificielle (adversarial search), avec expérimentation de plusieurs stratégies de décision sous contrainte de temps.