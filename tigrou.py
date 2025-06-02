import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

from core.puzzle import Puzzle
from core.puzzle import PuzzleSolver

import matplotlib as mp
from matplotlib import font_manager

# === Personnalisation de la police ===
# Spécifie ici le chemin vers ta police
font_path = "/Users/cyriaqueperier/Library/Fonts/cmunrm.ttf"  # <-- à adapter

# Ajoute la police et définis-la comme police par défaut
font_manager.fontManager.addfont(font_path)
custom_font = font_manager.FontProperties(fname=font_path).get_name()
mp.rcParams['font.family'] = custom_font

IMG_SRC = "/Users/cyriaqueperier/Pictures/IMG_4029.jpg"
FRAG_WIDTH = 100
FRAG_HEIGHT = 100
REDUCTION = 50

def new_random_permutation(n):
        
        L = [i for i in range(n)]
        
        return [L.pop(random.randint(0, n - 1 - i)) for i in range(n)]


image = np.array(PIL.Image.open(IMG_SRC).convert("L"))
print(image.shape)

start_height = (image.shape[0] // 2) - (image.shape[1] // 2)
image = image[start_height:start_height + image.shape[1]]
image = cv2.resize(image, (2000, 2000))

fragments = np.zeros((image.shape[0] // FRAG_WIDTH, image.shape[1] // FRAG_HEIGHT, FRAG_WIDTH, FRAG_HEIGHT), dtype=np.uint8)
Px = new_random_permutation(fragments.shape[0])
Py = new_random_permutation(fragments.shape[1])

for i in range(image.shape[0] // FRAG_WIDTH):
    for j in range(image.shape[1] // FRAG_HEIGHT):
        
        fragments[j, i] = image[i*FRAG_WIDTH:(i+1)*FRAG_WIDTH, j*FRAG_HEIGHT:(j+1)*FRAG_HEIGHT]

pattern = cv2.resize(image, (image.shape[0] // REDUCTION, image.shape[1] // REDUCTION))

copy = fragments[0, 0].copy()
fragments[0, 0] = fragments[1, 1].copy()
fragments[1, 1] = copy

permutation = np.arange(0, fragments.shape[0] * fragments.shape[1], 1).astype(np.uint16)
np.random.shuffle(permutation)
fragments = fragments.reshape((fragments.shape[0] * fragments.shape[1], ) + fragments.shape[2:])
for i in range(fragments.shape[0]):
    
    copy = fragments[i].copy()
    fragments[i] = fragments[permutation[i]].copy()
    fragments[permutation[i]] = copy

fragments = fragments.reshape((image.shape[0] // FRAG_WIDTH, image.shape[1] // FRAG_HEIGHT) + fragments.shape[1:])
puzzle = Puzzle(FRAG_HEIGHT, FRAG_WIDTH, fragments)
solver = PuzzleSolver(puzzle, pattern, process_now=False)

"""
plt.subplot(1, 3, 1)
plt.title("Image originale")
plt.imshow(image, cmap="gray")"""

plt.subplot(1, 2, 1)
plt.title("Modèle")
plt.imshow(pattern, cmap="gray")

"""
plt.subplot(1, 3, 3)
plt.title("Puzzle généré")
plt.imshow(puzzle.build(), cmap="gray")

"""
print(solver.solve())
print(solver.pixel_move_nbr)
print(int((solver.puzzle.fragments != fragments).sum()))

plt.subplot(1, 2, 2)
plt.title("Puzzle résolu")
plt.imshow(puzzle.build(), cmap="gray")

plt.tight_layout()
plt.savefig("/Users/cyriaqueperier/Documents/TIPE 2/Images/tigrou_erreur.jpeg", dpi=600.0, bbox_inches="tight")
plt.show()