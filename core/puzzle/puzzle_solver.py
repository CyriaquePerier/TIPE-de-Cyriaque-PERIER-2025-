#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 20:42:05 2024

@author: cyriaqueperier
"""

import os
import torch
import numpy as np
import scipy
import math
import PIL
from scipy.optimize import linear_sum_assignment
from skimage.feature import match_template

import cv2

from PIL import ImageFilter


class PuzzleSolver:
    """Résolveur de puzzle"""

    def __init__(
        self,
        puzzle,
        raw_pattern,
        force_bottom_position=None,
        force_right_position=None,
        edges_heuristic=False,
        debug=False,
        process_now=True,
        no_update_frag=False,
        prepare_pattern=True
    ):

        self.puzzle = puzzle
        self.raw_pattern = raw_pattern

        self.pattern_prepared = False
        self.cut = False
        self.reduced = False
        self.metrics_calculated = False
        self.deep_metrics_calculated = False

        self.debug = debug
        self.no_update_frag = no_update_frag
        self.process_now = process_now

        self.force_keep_position = np.zeros(
            self.puzzle.fragments.shape[:2], dtype=np.bool_
        )

        self.force_right_position = force_right_position
        self.force_bottom_position = force_bottom_position

        self.tiny_fragment_width = math.ceil(
            raw_pattern.shape[1] / puzzle.fragments.shape[0]
        )
        self.tiny_fragment_height = math.ceil(
            raw_pattern.shape[0] / puzzle.fragments.shape[1]
        )

        self.reduced_fragments = np.zeros(
            (
                puzzle.fragments.shape[0],
                puzzle.fragments.shape[1],
                self.tiny_fragment_height,
                self.tiny_fragment_width,
            )
        )
        self.fragments_pattern = np.zeros(
            (
                puzzle.fragments.shape[0],
                puzzle.fragments.shape[1],
                self.tiny_fragment_height,
                self.tiny_fragment_width,
            )
        )

        self.metrics = np.zeros(
            (
                self.reduced_fragments.shape[0],
                self.reduced_fragments.shape[1],
                self.reduced_fragments.shape[0],
                self.reduced_fragments.shape[1],
            ),
            dtype=np.float32,
        )

        self.pixel_error = 0

        self.fragment_move_nbr = 0
        self.pixel_move_nbr = 0

        # self.net = torch.load(os.getcwd() + "/DeepPatternMatcher.pt")
        # self.net.eval()

        if prepare_pattern:
            self.prepare_pattern()

        if self.process_now:
            self.cut_pattern()
            self.reduce_fragments()
            self.calculate_metrics()
            self.solve()
        # self.calculate_deep_metrics()

    def prepare_pattern(self, force=False):
        """Redimensionner le pattern de manière à éviter les problème d'alignement"""

        if (not self.pattern_prepared) or force:

            target_height = self.tiny_fragment_height * self.puzzle.fragments.shape[0]
            target_width = self.tiny_fragment_width * self.puzzle.fragments.shape[1]
            self.pattern = cv2.resize(self.raw_pattern, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

    def cut_pattern(self, force=False):
        """Découpe le plan en morceau de taille égale (et complète par du noir au besoin)"""

        if (not self.cut) or force:

            new_width = self.tiny_fragment_width * self.puzzle.fragments.shape[0]
            new_height = self.tiny_fragment_height * self.puzzle.fragments.shape[1]

            delta_h = new_height - self.raw_pattern.shape[0]
            delta_w = new_width - self.raw_pattern.shape[1]

            self.pixel_error = abs(delta_h) + abs(delta_w)

            if (
                new_width == self.raw_pattern.shape[1]
                and new_height == self.raw_pattern.shape[0]
            ):

                self.pattern = self.raw_pattern

            elif self.pixel_error == 200:

                self.pattern = np.pad(self.raw_pattern, ((0, delta_h), (0, delta_w)))

            else:

                self.pattern = cv2.resize(self.raw_pattern, (new_width, new_height))

            # Reshape en blocs
            for i in range(self.fragments_pattern.shape[0]):

                for j in range(self.fragments_pattern.shape[1]):

                    self.fragments_pattern[i, j] = self.pattern[
                        self.tiny_fragment_height
                        * j : self.tiny_fragment_height
                        * (j + 1),
                        self.tiny_fragment_width
                        * i : self.tiny_fragment_width
                        * (i + 1),
                    ]

            self.cut = True

    def cut_pattern_2(self, force=False):
        """Découpe le plan en morceau de taille égale (et complète par du noir au besoin)"""

        if (not self.cut) or force:
            new_width = self.tiny_fragment_width * self.puzzle.fragments.shape[0]
            new_height = self.tiny_fragment_height * self.puzzle.fragments.shape[1]

            delta_h = new_height - self.raw_pattern.shape[0]
            delta_w = new_width - self.raw_pattern.shape[1]

            self.pixel_error = abs(delta_h) + abs(delta_w)

            if (
                new_width == self.raw_pattern.shape[1]
                and new_height == self.raw_pattern.shape[0]
            ):
                self.pattern = self.raw_pattern
            elif self.pixel_error == 200:
                self.pattern = np.pad(self.raw_pattern, ((0, delta_h), (0, delta_w)))
            else:
                self.pattern = cv2.resize(self.raw_pattern, (new_width, new_height))

            # Reshape en blocs sans double boucle
            h, w = self.tiny_fragment_height, self.tiny_fragment_width
            fh, fw = self.puzzle.fragments.shape

            # Découpe et reshape
            self.pattern = self.pattern[
                : fh * h, : fw * w
            ]  # S'assurer des dimensions exactes
            self.fragments_pattern = self.pattern.reshape(fh, h, fw, w).swapaxes(1, 2)

            self.cut = True

    def cut_pattern_adaptative(self, force=False):
        """Découpe le plan en morceau de taille égale (et complète par du noir au besoin)"""

        if (not self.cut) or force:

            new_width = self.tiny_fragment_width * self.puzzle.fragments.shape[0]
            new_height = self.tiny_fragment_height * self.puzzle.fragments.shape[1]

            delta_h = new_height - self.raw_pattern.shape[0]
            delta_w = new_width - self.raw_pattern.shape[1]

            self.pixel_error = abs(delta_h)

            if self.pixel_error == 1 or self.pixel_error == 4:

                self.pattern = cv2.resize(self.raw_pattern, (new_width, new_height))

            else:

                self.pattern = np.pad(self.raw_pattern, ((0, 0), (0, delta_w)))

            # Reshape en blocs
            for i in range(self.fragments_pattern.shape[0]):

                for j in range(self.fragments_pattern.shape[1]):

                    height_start = min(
                        math.floor(
                            (self.pattern.shape[0] / self.puzzle.fragments.shape[1]) * j
                        ),
                        self.pattern.shape[0] - self.tiny_fragment_height,
                    )
                    height_stop = min(
                        math.ceil(
                            (self.pattern.shape[0] / self.puzzle.fragments.shape[1])
                            * (j + 1)
                        ),
                        self.pattern.shape[0],
                    )
                    frag = self.pattern[
                        height_start:height_stop,
                        self.tiny_fragment_width
                        * i:self.tiny_fragment_width
                        * (i + 1),
                    ]

                    self.fragments_pattern[i, j] = cv2.resize(
                        frag, (self.tiny_fragment_width, self.tiny_fragment_height)
                    )

            self.cut = True

    def reduce_fragments(self, force=False):
        """Réduit la qualité (et donc la taille) de chaque pièce du puzzle"""

        if (not self.reduced) or force:

            if self.debug:

                print("[puzzle_solver] Reducing fragments")

            for i in range(self.reduced_fragments.shape[0]):

                for j in range(self.reduced_fragments.shape[1]):

                    # pil_image = PIL.Image.fromarray(self.puzzle.fragments[i, j])
                    # self.reduced_fragments[i, j] = np.array(pil_image.resize((self.tiny_fragment_width, self.tiny_fragment_height)))
                    self.reduced_fragments[i, j] = cv2.resize(
                        self.puzzle.fragments[i, j],
                        (self.tiny_fragment_width, self.tiny_fragment_height),
                        interpolation=cv2.INTER_AREA,
                    )

            self.reduced = True

    def calculate_deep_metrics(self, force=False):
        """Calcule la distance entre les fragments du puzzle et du modèle avec un modèle PyTorch"""

        if (not self.deep_metrics_calculated) or force:

            data = np.zeros((2,) + 2 * self.reduced_fragments.shape[:2] + (5, 5))
            data[0, :, :] += np.pad(
                self.reduced_fragments,
                (
                    (0, 0),
                    (0, 0),
                    (0, 5 - self.reduced_fragments.shape[-2]),
                    (0, 5 - self.reduced_fragments.shape[-1]),
                ),
            )
            data[1, :, :] += np.pad(
                self.fragments_pattern,
                (
                    (0, 0),
                    (0, 0),
                    (0, 5 - self.fragments_pattern.shape[-2]),
                    (0, 5 - self.fragments_pattern.shape[-1]),
                ),
            )
            data = data.reshape((2, (data.shape[1] * data.shape[2]) ** 2, 5, 5))
            data = data.astype(dtype=np.float32)
            data = data.swapaxes(0, 1)

            torch_data = torch.from_numpy(data)
            print(torch_data.size())
            output = self.net(torch_data).max(1).indices
            output = output.reshape(2 * self.reduced_fragments.shape[:2])

            self.deep_metrics = 1000 * output.detach().numpy().astype(np.bool_)

            self.deep_metrics_calculated = True

    def calculate_metrics(self, force=False, edges_heuristic=False):
        """Calcule la distance entre les fragments du puzzle et du modèle"""

        if (not self.metrics_calculated) or force:

            # Conversion des fragments en dtype int16
            fragments_pattern_int16 = self.fragments_pattern.astype(np.int16)

            reduced_fragments_int16 = self.reduced_fragments.astype(np.int16)

            # Ajout de nouvelles dimensions pour permettre la diffusion
            diff = (
                fragments_pattern_int16[:, :, np.newaxis, np.newaxis, :, :]
                - reduced_fragments_int16[np.newaxis, np.newaxis, :, :, :, :]
            )

            if self.pixel_error < 2:

                w_avg = 0
                w_var = 2
                w_max = 0
                w_min = 0

                avg = np.sum(np.abs(diff), axis=(-2, -1)).astype(dtype=np.float64)
                avg /= round(np.average(avg))
                var = np.var(diff, axis=(-2, -1)).astype(dtype=np.float64)
                var /= round(np.average(var))

                current_metric = (
                    (w_avg * avg)
                    + (w_var * var)
                    + (w_max * np.max(diff, axis=(-2, -1)).astype(dtype=np.int16))
                    + (w_min * np.min(diff, axis=(-2, -1)).astype(dtype=np.int16))
                )
            else:

                w_avg = 0
                w_var = 2
                w_max = 0
                w_min = 0

                avg = np.sum(diff, axis=(-2, -1)).astype(dtype=np.float64)
                avg /= round(np.average(avg))
                var = np.var(diff, axis=(-2, -1)).astype(dtype=np.float64)
                var /= round(np.average(var))

                current_metric = (
                    (w_avg * avg)
                    + (w_var * var)
                    + (w_max * np.max(diff, axis=(-2, -1)).astype(dtype=np.int16))
                    + (w_min * np.min(diff, axis=(-2, -1)).astype(dtype=np.int16))
                )

            self.metrics = (1000 * current_metric).round()

            # Heuristique relative à la bordure

            if edges_heuristic:

                force_bottom_matrix = (
                    np.ones(
                        (
                            self.reduced_fragments.shape[0],
                            self.reduced_fragments.shape[1],
                        ),
                        dtype=np.uint16,
                    )
                    * 2
                )
                force_bottom_matrix[-1, :] = 1

                force_right_matrix = (
                    np.ones(
                        (
                            self.reduced_fragments.shape[0],
                            self.reduced_fragments.shape[1],
                        ),
                        dtype=np.uint16,
                    )
                    * 2
                )
                force_right_matrix[:, -1] = 1

                for i in range(self.reduced_fragments.shape[0]):

                    for j in range(self.reduced_fragments.shape[1]):

                        if self.force_bottom_position[i, j]:

                            self.metrics[i, j, :, :-1] = 100000000

                        if self.force_right_position[i, j]:

                            self.metrics[i, j, :-1, :] = 100000000

            self.metrics_calculated = True

    def solve(self, algorithm="hungarian"):
        """Résoudre le puzzle"""

        self.cut_pattern_adaptative()
        self.reduce_fragments()
        self.calculate_metrics()

        if algorithm == "gale_shapley":
            solution = self.solve_gale_shapley()
        elif algorithm == "greedy":
            solution = self.solve_greedy()
        else:
            solution = self.solve_hungarian()

        return solution

    def solve_hungarian(self):
        """Résoudre le puzzle en utilisant l'algorithme hongrois pour une affectation optimale"""

        # On suppose que self.metrics est une matrice 4D (MxN)x(PxQ)
        M, N, P, Q = self.metrics.shape

        # Reshape pour transformer la métrique en une matrice 2D de taille (M*N, P*Q)
        cost_matrix = self.metrics.reshape(M * N, P * Q)

        # Utiliser l'algorithme hongrois pour résoudre le problème d'affectation
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)

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
        self.pixel_move_nbr = int((self.puzzle.fragments != new_fragments).sum())
        self.puzzle.fragments = new_fragments

        return solution

    def solve_greedy(self):
        """Résoudre le puzzle avec l'algortithme de Kruskal"""

        # Récupération des indices
        indices = np.indices(self.metrics.shape)

        # Reformez les indices pour qu'ils aient la forme (nombre d'éléments, 4)
        indices = indices.reshape(4, -1).T

        # Fusion et tri des matrices
        metrics = np.column_stack((self.metrics.ravel(), indices))
        sorted_metrics = metrics[metrics[:, 0].argsort()]

        # Initialisation du tableau solution
        solution = np.zeros(
            (self.metrics.shape[0], self.metrics.shape[1], 2), dtype=np.uint16
        )
        is_solved_1 = np.zeros((metrics.shape[0], metrics.shape[1]), dtype=np.bool_)
        is_solved_2 = np.zeros((metrics.shape[0], metrics.shape[1]), dtype=np.bool_)
        new_fragments = np.zeros_like(self.puzzle.fragments)

        for index in range(sorted_metrics.shape[0]):

            x1, y1, x2, y2 = sorted_metrics[index, 1:5]
            x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

            if not (is_solved_1[x1, y1] or is_solved_2[x2, y2]):

                solution[x1, y1, 0] = x2
                solution[x1, y1, 1] = y2
                is_solved_1[x1, y1] = True
                is_solved_2[x2, y2] = True
                new_fragments[x1, y1] = self.puzzle.fragments[x2, y2]

        self.puzzle.fragments = new_fragments
        return solution

    def solve_gale_shapley(self):
        """Résoudre le puzzle en utilisant l'algorithme de Gale-Shapley pour un mariage stable"""

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
                if (
                    match_model_to_puzzle[model_idx] == -1
                ):  # Si le modèle n'est pas encore apparié
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
                        current_preference = np.where(
                            preferences_puzzle[puzzle_idx] == current_model
                        )[0][0]
                        new_preference = np.where(
                            preferences_puzzle[puzzle_idx] == model_idx
                        )[0][0]

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
