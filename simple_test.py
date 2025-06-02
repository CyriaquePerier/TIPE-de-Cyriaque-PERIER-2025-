import PIL.Image
import numpy as np
import cv2
import os
import sqlite3
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.puzzle import Puzzle, PuzzleSolver

DBB_SRC = "/Users/cyriaqueperier/Documents/TIPE 2/bdd_yggdrasil.db"
IMG_DIR_SRC = "/Users/cyriaqueperier/Documents/TIPE 2/Banque d'Image 2"
FRAG_NBR_HEIGHT = 21
FRAG_NBR_WIDTH = 21
REDUCT_RATE = 20


# Chargement des images
images = []
for name in os.listdir(IMG_DIR_SRC):
    if name.lower().endswith((".jpg", ".webp", ".png", ".jpeg")):
        print(f"Chargement de l'image {name}...")
        img = np.array(PIL.Image.open(os.path.join(IMG_DIR_SRC, name)).convert("L"))
        if img.shape[0] > 700 and img.shape[1] > 700:
            images.append(img)

image = images[0]
frag_h = image.shape[0] // FRAG_NBR_HEIGHT
frag_w = image.shape[1] // FRAG_NBR_WIDTH
img_resized = cv2.resize(image, (frag_w * FRAG_NBR_WIDTH, frag_h * FRAG_NBR_HEIGHT))

fragments = np.zeros((FRAG_NBR_HEIGHT, FRAG_NBR_WIDTH, frag_h, frag_w), dtype=np.uint8)
for i in range(FRAG_NBR_WIDTH):
    for j in range(FRAG_NBR_HEIGHT):
        fragments[j, i] = img_resized[i*frag_h:(i+1)*frag_h, j*frag_w:(j+1)*frag_w]

puzzle = Puzzle(frag_w, frag_h, fragments)
pattern = cv2.resize(img_resized,
                        (img_resized.shape[1] // REDUCT_RATE,
                        img_resized.shape[0] // REDUCT_RATE))
solver = PuzzleSolver(puzzle, pattern, debug=False, no_update_frag=False, process_now=True, prepare_pattern=True)
PIL.Image.fromarray(puzzle.build()).show()