#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt 
import matplotlib as mp
from matplotlib import font_manager
from collections import deque
import seaborn as sns


# Réglage de Matplotlib pour l'exportation des données
FONT_PATH = "/Users/cyriaqueperier/Library/Fonts/cmunrm.ttf"
font_manager.fontManager.addfont(FONT_PATH)
custom_font = font_manager.FontProperties(fname=FONT_PATH).get_name()
mp.rcParams['font.family'] = custom_font
mp.rcParams["font.size"] = 20.0


class GeneralCasePuzzleSolver:
    """ Tentative initiale de résolution de puzzle dans le cas général """


    def __init__(self, puzzle, process_now=True):
        """ Initialisation de l'objet et définition des paramètres """

        self.fragments = puzzle.fragments
        self.puzzle = puzzle

        self.top_bot_metrics = None
        self.left_right_metrics = None

        i_indices, j_indices = np.meshgrid(np.arange(self.fragments.shape[0]), np.arange(self.fragments.shape[1]), indexing='ij')
        self.current_solution = np.stack((i_indices, j_indices), axis=-1)
        # CONVENTION: Le fragment en position [i, j] sur la grille est le fragment self.current_solution[i, j] dans self.fragments

        self.metrics_calculated = False

        if process_now:

            self.calculate_metrics()

    def calculate_metrics(self, force=False):
        """ Calcul des distances """

        if (not self.metrics_calculated) or force:

            last_cols = self.fragments[:, :, :, -1]
            first_cols = self.fragments[:, :, :, 0]
            last_cols_exp = last_cols[:, :, None, None, :]
            first_cols_exp = first_cols[None, None, :, :, :]
            self.top_bot_metrics = np.sum(np.abs(last_cols_exp - first_cols_exp), axis=-1)

            last_rows = self.fragments[:, :, -1, :]
            first_rows = self.fragments[:, :, 0, :]
            last_rows_exp = last_rows[:, :, None, None, :]
            first_rows_exp = first_rows[None, None, :, :, :]
            self.left_right_metrics = np.sum(np.abs(last_rows_exp - first_rows_exp), axis=-1)

            self.metrics_calculated = True
    
    def calculate_weight(self, external_solution=None):
        """ Calcul le poids d'une solution """

        solution = external_solution
        if external_solution is None:
            solution = self.current_solution

        weight = 0

        for i in range(solution.shape[0] - 1):
            for j in range(solution.shape[1] - 1):

                # droite
                weight += self.left_right_metrics[
                    solution[i, j, 0], solution[i, j, 1],
                    solution[i, j+1, 0], solution[i, j+1, 1]
                ]

                # bas
                weight += self.top_bot_metrics[
                    solution[i, j, 0], solution[i, j, 1],
                    solution[i+1, j, 0], solution[i+1, j, 1]
                ]
        
        return weight
        
    def extract_statistics(self, display=True, save_file=None):
        """ Génère un graphique autour de la distribution des valeurs prises par les normes calculées """

        if not self.metrics_calculated:
            self.calculate_metrics

        size = (self.left_right_metrics.shape[0] * self.left_right_metrics.shape[1]) ** 2 # Nombre de distances horizontales (idem pour les distances verticales)
        array = np.concatenate((self.left_right_metrics.reshape((size)), self.top_bot_metrics.reshape((size))))

        stats_dict = {
            "Moyenne": np.mean(array),
            "Médiane": np.median(array),
            "Écart-type": np.std(array),
        }

        plt.figure(figsize=(10, 5))
        sns.histplot(array, kde=True, bins='auto', color='skyblue', edgecolor='black')
        plt.title("Distribution des normes entre deux fragments de puzzle")
        plt.xlabel("Norme")
        plt.ylabel("Fréquence")
        plt.grid(True)
        plt.tight_layout()
        
        if display:
            plt.show()
        
        if not (save_file is None):

            plt.savefig(save_file, dpi=300.0)

    def apply_solution(self, solution):
        """ Applique une solution à l'objet puzzle enregistré (attention: cette classe devient inutilisable) """

        fragments = np.zeros_like(self.fragments)

        for i in range(solution.shape[0]):
            for j in range(solution.shape[1]):

                fragments[i, j] = self.fragments[solution[i, j, 0], solution[i, j, 1]]
        
        self.puzzle.fragments = fragments

    def solve_greedy(self, start_pos, start_fragment, apply_solution=True):
        """ Résout le puzzle par algorithme glouton en partant du fragment `start_fragment` placé en `start_pos` """

        if not self.metrics_calculated:
            self.calculate_metrics()

        n, m = self.fragments.shape[:2]

        # Grille résultat : chaque case contiendra un tuple (k, l)
        solution = np.full((n, m, 2), fill_value=-1, dtype=int)
        used = np.zeros((n, m), dtype=bool)

        # Initialisation
        pi, pj = start_pos
        fi, fj = start_fragment
        solution[pi, pj] = (fi, fj)
        used[fi, fj] = True
        queue = deque()
        queue.append((pi, pj))

        # Directions : (dy, dx, metric)
        directions = [
            (-1,  0, 'top_bot'),   # haut
            ( 1,  0, 'top_bot'),   # bas
            ( 0, -1, 'left_right'),  # gauche
            ( 0,  1, 'left_right')   # droite
        ]

        while queue:
            i, j = queue.popleft()
            frag_i, frag_j = solution[i, j]

            for di, dj, metric_type in directions:
                ni, nj = i + di, j + dj

                # Vérifie que la position est dans la grille et encore vide
                if 0 <= ni < n and 0 <= nj < m and (solution[ni, nj] == -1).all():
                    best_score = float('inf')
                    best_fragment = None

                    # Cherche le meilleur fragment non utilisé
                    for cand_i in range(n):
                        for cand_j in range(m):
                            if used[cand_i, cand_j]:
                                continue

                            if metric_type == 'left_right':
                                if dj == 1:
                                    # Droite de (i, j) contre gauche du candidat
                                    score = self.left_right_metrics[frag_i, frag_j, cand_i, cand_j]
                                else:
                                    # Gauche de (i, j) contre droite du candidat
                                    score = self.left_right_metrics[cand_i, cand_j, frag_i, frag_j]
                            else:  # top_bot
                                if di == 1:
                                    # Bas de (i, j) contre haut du candidat
                                    score = self.top_bot_metrics[frag_i, frag_j, cand_i, cand_j]
                                else:
                                    # Haut de (i, j) contre bas du candidat
                                    score = self.top_bot_metrics[cand_i, cand_j, frag_i, frag_j]

                            if score < best_score:
                                best_score = score
                                best_fragment = (cand_i, cand_j)

                    if best_fragment:
                        solution[ni, nj] = best_fragment
                        used[best_fragment] = True
                        queue.append((ni, nj))

        self.current_solution = solution

        if apply_solution:
            self.apply_solution(solution)

        return solution
