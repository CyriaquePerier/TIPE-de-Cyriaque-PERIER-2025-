#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:31:00 2024

@author: cyriaqueperier
"""

import numpy as np

class Puzzle:
    
    """ Représentation d'un puzzle """
    
    def __init__(self, hmp, wmp, fragments):
        
        """ Initialisation de l'objet ."""
        
        self.hmp = hmp # Hauteur maximale en pixel d'un fragment
        self.wmp = wmp # Largeur maximale en pixel d'un fragment
        self.fragments = fragments # Tableau numpy de format (x, y, wmp, hmp) avec x et y respectivement la largeur et la hauteur en fragment du puzzle
        self.archive = {}
        
    def build(self):
        
        """ Construit l'image finale en assemblant les fragments. """
        
        width = self.hmp * self.fragments.shape[0]
        height = self.wmp * self.fragments.shape[1]
        
        image = np.zeros((width, height))
        L = []
        
        for i in range(self.fragments.shape[0]):
            
            L.append(np.vstack(self.fragments[i]))
            
        return np.hstack(L)

if __name__ == "__main__":
    
    import json
    import PIL
    import io
    import base64
    
    def to_square_50(matrix):
        """Convertit un tableau numpy rectangulaire en un carré de taille 50 en remplissant les vides avec des zéros."""
        # Obtenir les dimensions du tableau d'entrée
        original_height, original_width = matrix.shape
        
        # Vérifier que les dimensions sont inférieures à 50
        if original_height > 50 or original_width > 50:
            raise ValueError("Les dimensions du tableau doivent être inférieures ou égales à 50")
        
        # Calculer les quantités de remplissage nécessaires
        pad_height = 50 - original_height
        pad_width = 50 - original_width
        
        # Ajouter des zéros pour convertir en carré de taille 50
        padded_matrix = np.pad(matrix, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
        
        return padded_matrix
    
    file = open("/Users/cyriaqueperier/Documents/DocScraper/test_answer.json", "r")
    json_content = json.load(file)
    file.close()

    raw_images = [base64.b64decode(json_content["images"][i]) for i in range(len(json_content["images"]))]
    images = [to_square_50(np.array(PIL.Image.open(io.BytesIO(raw_image)))) for raw_image in raw_images]
    
    l = len(images) // 3
    fragments = np.zeros((l, 3, images[0].shape[0], images[1].shape[1]), dtype=np.uint8)
    for i in range(3):
        fragments[:, i] = np.array(images[l * i:l * (i + 1)])
    
    puzzle = Puzzle(50, 50, fragments)
    np_image = puzzle.build()
    pillow_image = PIL.Image.fromarray(np_image)
    pillow_image.show()
    
    