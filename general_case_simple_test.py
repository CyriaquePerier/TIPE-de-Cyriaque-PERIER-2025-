import PIL.Image
import numpy as np
import cv2
import os
from core.puzzle import Puzzle, GeneralCasePuzzleSolver

DBB_SRC = "/Users/cyriaqueperier/Documents/TIPE 2/bdd_yggdrasil.db"
IMG_DIR_SRC = "/Users/cyriaqueperier/Documents/TIPE 2/Banque d'Image 2"
FRAG_NBR_HEIGHT = 5
FRAG_NBR_WIDTH = 5


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
solver = GeneralCasePuzzleSolver(puzzle)
print(solver.calculate_weight())
solver.extract_statistics()
solver.solve_greedy((0, 0), (0, 0))
print(solver.calculate_weight())

PIL.Image.fromarray(puzzle.build()).show()
