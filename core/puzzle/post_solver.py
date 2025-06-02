# -*- coding: utf-8 -*-

from collections import deque
import numpy as np
import cv2

class PostSolver:

    def __init__(self, puzzle, pattern) -> None:
        
        """ Initialisation de l'objet """

        # Stockage des variables utiles
        self.puzzle = puzzle
        self.pattern = pattern.astype(np.int16)

        # Initalisation de la file de priorité des fragments à permutter 
        self.change_priority = deque()
        
    def generate_change_priority(self) -> None:

        """ Générer la file de priorité des fragments à permutter """

        # Pour l'instant, on ajoute bêtement tous les fragments à la file (à optimiser)
        for i in range(self.puzzle.fragments.shape[0]):

            for j in range(self.puzzle.fragments.shape[1]):

                self.change_priority.append((i, j))

    def change_fragment(self) -> None:

        """ Déplacer un fragment en une position optimisant le coût actuel du puzzle par rapport au modèle """

        # On récupère les coordonées fragment à déplacer en priorité
        frag_i, frag_j = self.change_priority.popleft()
        best_change_i, best_change_j = frag_i, frag_j
        best_change_cost = self.calculate_cost()
        p = best_change_cost
        for i, j in self.change_priority:

            fragments = self.puzzle.fragments.copy()
            # Échange temporaire des fragments
            change = fragments[frag_i, frag_j].copy()
            fragments[frag_i, frag_j] = fragments[i, j].copy()
            fragments[i, j] = change
            
            current_cost = self.calculate_cost(fragments)

            if current_cost < best_change_cost:  # Vérifie si le coût est amélioré
                best_change_cost = current_cost
                best_change_i = i
                best_change_j = j

        # On effectue le changement optimal
        change = self.puzzle.fragments[frag_i, frag_j].copy()
        self.puzzle.fragments[frag_i, frag_j] = self.puzzle.fragments[best_change_i, best_change_j].copy()
        self.puzzle.fragments[best_change_i, best_change_j] = change

        print("{} <-> {} ({} -> {})".format((best_change_i, best_change_j), (frag_i, frag_j), p, best_change_cost))
        
        if best_change_i != frag_i or best_change_j != frag_j:
            self.change_priority.append((frag_i, frag_j))

    def calculate_cost(self, fragments=None) -> int:
        
        """ Calculer le coût actuel du puzzle par rapport au modèle """

        if fragments is None:

            fragments = self.puzzle.fragments

        width = self.puzzle.wmp * fragments.shape[0]
        height = self.puzzle.wmp * fragments.shape[1]
        L = []
        
        for i in range(fragments.shape[0]):
            
            L.append(np.vstack(fragments[i]))
            
        puzzle_image = cv2.resize(np.hstack(L), (self.pattern.shape[1], self.pattern.shape[0]), interpolation=cv2.INTER_AREA).astype(np.int16)

        return np.abs(self.pattern - puzzle_image).sum()
    
    def post_solve(self) -> None:

        """ Optimise la solution déjà proposée """

        self.generate_change_priority()
        while len(self.change_priority) > 0:
            self.change_fragment()