import json
import ast
import os
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from oxono_net import OxonoNet, encode_state


class OxonoDataset(Dataset):
    def __init__(self, fichier):
        with open(fichier) as f:
            self.data = json.load(f)
        if not isinstance(self.data, list) or len(self.data) == 0:
            raise ValueError(f"Dataset vide ou invalide: {fichier}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entree = self.data[idx]

        board_value = entree.get("board")
        if isinstance(board_value, str):
            board = ast.literal_eval(board_value)
        else:
            board = board_value

        # Compatibilite: ancien format (joueur/resultat) et variantes eventuelles.
        joueur = entree.get("joueur", entree.get("player"))
        resultat_value = entree.get("resultat", entree.get("result"))
        if joueur is None or resultat_value is None:
            raise KeyError("Chaque entree doit contenir 'joueur'/'player' et 'resultat'/'result'")

        tenseur = encode_state(board, int(joueur))
        resultat = torch.tensor(float(resultat_value), dtype=torch.float32)

        return tenseur.squeeze(0), resultat


def entrainer(
    nb_epochs=10,
    batch_size=64,
    fichier="dataset_train.json",
    learning_rate=0.001,
    output_path="modeles/oxono_net.pth",
):
    dataset = OxonoDataset(fichier)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    os.makedirs("modeles", exist_ok=True)

    reseau = OxonoNet()
    optimizer = torch.optim.Adam(reseau.parameters(), lr=learning_rate)
    critere = nn.MSELoss()

    print(f"Dataset: {fichier}")
    print(f"Nb etats: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")

    for epoch in range(nb_epochs):
        perte_totale = 0

        for tenseurs, resultats in dataloader:
            predictions  = reseau(tenseurs)
            perte        = critere(predictions, resultats)

            optimizer.zero_grad()
            perte.backward()
            optimizer.step()

            perte_totale += perte.item()

        print(f"Epoch {epoch+1}/{nb_epochs} — perte: {perte_totale/len(dataloader):.6e}")
        torch.save(reseau.state_dict(), f"modeles/oxono_net_epoch_{epoch+1}.pth")

    # Sauvegarde des poids
    torch.save(reseau.state_dict(), output_path)
    print(f"Modele sauvegarde: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Entrainement du reseau Oxono")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset_train.json",
        help="Fichier JSON de dataset (ex: dataset_train.json)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Nombre d'epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Taille de batch",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate Adam",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="modeles/oxono_net.pth",
        help="Chemin du modele final",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed aleatoire pour reproductibilite",
    )
    return parser.parse_args()


def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    entrainer(
        nb_epochs=args.epochs,
        batch_size=args.batch_size,
        fichier=args.dataset,
        learning_rate=args.lr,
        output_path=args.output,
    )