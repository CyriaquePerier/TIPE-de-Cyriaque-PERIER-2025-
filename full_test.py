#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import PIL.Image
import numpy as np
import cv2
import os
import sqlite3
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.puzzle import Puzzle, PuzzleSolver

DBB_SRC = "/Users/cyriaqueperier/Documents/TIPE 2/bdd_yggdrasil.db"
IMG_DIR_SRC = "/Users/cyriaqueperier/Documents/TIPE 2/Banque d'Image"
MAX_FRAG_NBR = 21
MAX_REDUCT_RATE = 51



# Chargement des images
images = []
for name in os.listdir(IMG_DIR_SRC):
    if name.lower().endswith((".jpg", ".webp", ".png", ".jpeg")):
        print(f"Chargement de l'image {name}...")
        img = np.array(PIL.Image.open(os.path.join(IMG_DIR_SRC, name)).convert("L"))
        if img.shape[0] > 700 and img.shape[1] > 700:
            images.append(img)

# Traitement parallèle avec ThreadPoolExecutor

def process_config(np_image, num_frag, reduct_rate):
    print("Processing {} fragments with reduction rate {}/{}...".format(num_frag, reduct_rate, MAX_REDUCT_RATE))

    img_resized = np_image
    frag_h = img_resized.shape[0] // num_frag
    frag_w = img_resized.shape[1] // num_frag
    img_resized = cv2.resize(img_resized, (frag_w * num_frag, frag_h * num_frag))

    # Découpage des fragments
    fragments = np.zeros((num_frag, num_frag, frag_h, frag_w), dtype=np.uint8)
    for i in range(num_frag):
        for j in range(num_frag):
            fragments[j, i] = img_resized[i*frag_h:(i+1)*frag_h, j*frag_w:(j+1)*frag_w]

    puzzle = Puzzle(frag_w, frag_h, fragments)
    pattern = cv2.resize(img_resized,
                         (img_resized.shape[1] // reduct_rate,
                          img_resized.shape[0] // reduct_rate))
    solver = PuzzleSolver(puzzle, pattern, debug=False, no_update_frag=True, process_now=False)
    solver.solve()
    return (img_resized.shape[0], img_resized.shape[1], num_frag, reduct_rate, solver.pixel_move_nbr)

# Connexion BDD
conn = sqlite3.connect(DBB_SRC)
cursor = conn.cursor()
# cursor.execute("CREATE TABLE IF NOT EXISTS results(image_width, image_height, num_frag, reduct_rate, move_nbr)")

for np_image in images:
    # Pour chaque image, on peut paralléliser num_frag × reduct_rate
    tasks = []
    with ThreadPoolExecutor() as executor:
        for num_frag in range(5, MAX_FRAG_NBR):
            for reduct_rate in range(5, MAX_REDUCT_RATE):
                if np_image.shape[0] >= reduct_rate and np_image.shape[1] >= reduct_rate:
                    tasks.append(executor.submit(process_config, np_image, num_frag, reduct_rate))

        # Collecte et insertion des résultats
        results = []
        for future in as_completed(tasks):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Erreur lors du traitement: {e}")

    # Batch insert
    cursor.executemany(
        "INSERT INTO results_nolimit VALUES (?, ?, ?, ?, ?)",
        results
    )
    conn.commit()

conn.close()
