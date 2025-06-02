#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2

class GPMSolver:

    def __init__(self, puzzle, pattern) -> None:
        
        # Sauvegarde des données utiles
        self.puzzle = puzzle
        self.pattern = pattern

        # Initialisation du tableau des distances
        self.metrics = np.zeros(2 * self.puzzle.fragments.shape[:2], dtype=np.uint32)

        # Génération du modèle de taille élevée
        large_pattern_size = (self.puzzle.fragments.shape[0] * self.puzzle.fragments.shape[2], self.puzzle.fragments.shape[1] * self.puzzle.fragments.shape[3])
        self.large_pattern = cv2.resize(self.pattern, puzzle.build().shape, interpolation=cv2.INTER_AREA)
        print(large_pattern_size)
        print(self.large_pattern.shape)

        self.metrics_calculated = False
        self.solved = False

        self.calculate_metrics()

    def create_test_pattern(self, fragment, i, j):

        """ Créer un modèle de taille élevée incluant un fragment du puzzle """

        #print(str(i) + " " + str(j))

        test_pattern = self.large_pattern.copy()
        test_pattern[fragment.shape[0] * j : fragment.shape[0] * (j + 1), fragment.shape[1] * i : fragment.shape[1] * (i + 1)] = fragment

        return test_pattern

    def calculate_metrics(self, force=False):
        
        """ Calcule la distance entre les fragments du puzzle et du modèle """
        
        
        if (not self.metrics_calculated) or force:
            
            test_patterns = np.zeros(self.metrics.shape + self.pattern.shape, dtype=np.int64)
            print(test_patterns.shape)

            for i in range(test_patterns.shape[0]):

                for j in range(test_patterns.shape[1]):

                    for k in range(test_patterns.shape[2]):

                        for l in range(test_patterns.shape[3]):

                            test_patterns[i, j, k, l] = cv2.resize(self.create_test_pattern(self.puzzle.fragments[i, j], k, l), (self.pattern.shape[1], self.pattern.shape[0]), interpolation=cv2.INTER_AREA)
            
            print("Calcul des distances...")

            diff = cv2.resize

            test_patterns[:, :, :, :] -= self.pattern
            test_patterns = np.abs(test_patterns)

            w_avg = 1
            w_var = 1
            w_max = 0
            w_min = 0
            
            avg = np.sum(test_patterns, axis=(-2, -1)).astype(dtype=np.float64)
            avg /= np.average(avg)
            var = np.var(test_patterns, axis=(-2, -1)).astype(dtype=np.float64)
            var /= np.average(var)
            
            current_metric = (w_avg * avg) + (w_var * var) + (w_max * np.max(test_patterns, axis=(-2, -1)).astype(dtype=np.int16)) + (w_min * np.min(test_patterns, axis=(-2, -1)).astype(dtype=np.int16))
            
            self.metrics = (1000 * current_metric).round()

            print("Distances calculées")
            self.metrics_calculated = True

    def solve(self, algorithm="hungarian"):
        
        """ Résoudre le puzzle """
        
        self.calculate_metrics()
        
        if algorithm == "gale_shapley": 
            solution = self.solve_gale_shapley()
        elif algorithm == "greedy":
            solution = self.solve_greedy()
        else:
            solution = self.solve_hungarian()
        
        return solution

    def solve_hungarian(self):
        
        """ Résoudre le puzzle en utilisant l'algorithme hongrois pour une affectation optimale """
    
        # On suppose que self.metrics est une matrice 4D (MxN)x(PxQ)
        M, N, P, Q = self.metrics.shape
        
        # Reshape pour transformer la métrique en une matrice 2D de taille (M*N, P*Q)
        cost_matrix = self.metrics.reshape(M * N, P * Q)
        
        # Utiliser l'algorithme hongrois pour résoudre le problème d'affectation
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Initialisation du tableau solution
        solution = np.zeros((M, N, 2), dtype=np.uint16)
        new_fragments = np.zeros_like(self.puzzle.fragments)
        
        # Reformuler les indices pour retrouver les positions dans les matrices d'origine
        for idx in range(len(row_ind)):
            model_idx = row_ind[idx]
            puzzle_idx = col_ind[idx]
            
            model_x, model_y = divmod(model_idx, N)
            puzzle_x, puzzle_y = divmod(puzzle_idx, Q)
            
            solution[model_x, model_y] = [puzzle_x, puzzle_y]
            self.puzzle.archive[(puzzle_x, puzzle_y)] = (model_x, model_y)
            new_fragments[model_x, model_y] = self.puzzle.fragments[puzzle_x, puzzle_y]
        
        # Mettre à jour les fragments du puzzle avec la solution trouvée
        self.puzzle.fragments = new_fragments
        
        return solution
    
    def solve_gale_shapley(self):
        
        """ Résoudre le puzzle en utilisant l'algorithme de Gale-Shapley pour un mariage stable """
    
        M, N, P, Q = self.metrics.shape
    
        # Reshape de la métrique en une matrice 2D de taille (M*N, P*Q)
        cost_matrix = self.metrics.reshape(M * N, P * Q)
    
        # Préférences des modèles : pour chaque fragment de modèle, classer les fragments de puzzle par affinité croissante
        preferences_model = np.argsort(cost_matrix, axis=1)
    
        # Préférences des puzzles : pour chaque fragment de puzzle, classer les fragments de modèle par affinité croissante
        preferences_puzzle = np.argsort(cost_matrix.T, axis=1)
    
        # Initialisation des appariements (valeur -1 signifie "non apparié")
        match_model_to_puzzle = -np.ones(M * N, dtype=int)
        match_puzzle_to_model = -np.ones(P * Q, dtype=int)
        
        # Stocker la proposition actuelle pour chaque fragment de modèle
        next_proposal = np.zeros(M * N, dtype=int)
    
        while np.any(match_model_to_puzzle == -1):
            for model_idx in range(M * N):
                if match_model_to_puzzle[model_idx] == -1:  # Si le modèle n'est pas encore apparié
                    # Obtenir le prochain puzzle à qui ce modèle va proposer
                    puzzle_idx = preferences_model[model_idx, next_proposal[model_idx]]
                    next_proposal[model_idx] += 1
    
                    # Si ce puzzle n'est pas encore apparié, accepter la proposition
                    if match_puzzle_to_model[puzzle_idx] == -1:
                        match_model_to_puzzle[model_idx] = puzzle_idx
                        match_puzzle_to_model[puzzle_idx] = model_idx
                    else:
                        # Si le puzzle est déjà apparié, vérifier s'il préfère ce modèle
                        current_model = match_puzzle_to_model[puzzle_idx]
                        current_preference = np.where(preferences_puzzle[puzzle_idx] == current_model)[0][0]
                        new_preference = np.where(preferences_puzzle[puzzle_idx] == model_idx)[0][0]
    
                        # Si le puzzle préfère ce nouveau modèle
                        if new_preference < current_preference:
                            # Swap des appariements
                            match_model_to_puzzle[current_model] = -1
                            match_model_to_puzzle[model_idx] = puzzle_idx
                            match_puzzle_to_model[puzzle_idx] = model_idx
    
        # Reformuler les indices pour les dimensions initiales
        solution = np.zeros((M, N, 2), dtype=np.uint16)
        new_fragments = np.zeros_like(self.puzzle.fragments)
    
        for model_idx, puzzle_idx in enumerate(match_model_to_puzzle):
            model_x, model_y = divmod(model_idx, N)
            puzzle_x, puzzle_y = divmod(puzzle_idx, Q)
    
            solution[model_x, model_y] = [puzzle_x, puzzle_y]
            new_fragments[model_x, model_y] = self.puzzle.fragments[puzzle_x, puzzle_y]
    
        self.puzzle.fragments = new_fragments
    
        return solution