from agent import Agent
from oxono import Game, State
import time
import random
import torch
import torch.nn as nn
import numpy as np

class OxonoNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Corps partagé : extrait les patterns du plateau
        self.corps = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  
            # 3 plans en entrée :
            #   plan 0 : cases de root_player (1 ou 0)
            #   plan 1 : cases de l'opponent  (1 ou 0)
            #   plan 2 : symbole de la pièce  (1=x, 0=o, 0=vide)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Tête value : produit un score entre -1 et +1
        self.value = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()   # borne la sortie dans [-1, +1]
        )

    def forward(self, x):
        x = self.corps(x)
        return self.value(x).squeeze(-1)
    
def encode_state(board : list[list], root_player : int):
    #opponent = 1 - root_player

    plan_root    = np.zeros((6, 6), dtype=np.float32)
    plan_opp     = np.zeros((6, 6), dtype=np.float32)
    plan_symbole = np.zeros((6, 6), dtype=np.float32)

    for i in range(6):
        for j in range(6):
            piece = board[i][j]
            if piece is not None:
                symbole, joueur = piece
                if joueur == root_player:
                    plan_root[i][j] = 1.0
                else:
                    plan_opp[i][j] = 1.0
                if symbole == "x":
                    plan_symbole[i][j] = 1.0

    # forme finale : (1, 3, 6, 6)
    # le 1 est le batch_size attendu par PyTorch
    tenseur = np.stack([plan_root, plan_opp, plan_symbole])
    return torch.tensor(tenseur).unsqueeze(0)