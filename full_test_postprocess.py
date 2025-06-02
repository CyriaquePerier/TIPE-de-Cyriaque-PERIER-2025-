#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mp
from matplotlib import font_manager

# === Personnalisation de la police ===
# Spécifie ici le chemin vers ta police
font_path = "/Users/cyriaqueperier/Library/Fonts/cmunrm.ttf"  # <-- à adapter

# Ajoute la police et définis-la comme police par défaut
font_manager.fontManager.addfont(font_path)
custom_font = font_manager.FontProperties(fname=font_path).get_name()
mp.rcParams['font.family'] = custom_font
mp.rcParams["font.size"] = 20.0

# === Connexion à la base de données ===
conn = sqlite3.connect("/Users/cyriaqueperier/Documents/TIPE 2/bdd_yggdrasil.db")
query = "SELECT num_frag, reduct_rate, image_width, image_height, move_nbr FROM results_backup WHERE num_frag <= 20 AND reduct_rate <= 20"
df = pd.read_sql_query(query, conn)
conn.close()

# === Calcul du ratio déplacé ===
df["total_pixels"] = df["image_width"] * df["image_height"]
df["ratio"] = df["move_nbr"] / df["total_pixels"]

# Moyenne des ratios pour chaque couple (num_frag, reduct_rate)
pivot = df.groupby(["num_frag", "reduct_rate"])["ratio"].mean().unstack()

# === Tracé de la heatmap ===
plt.figure(figsize=(12, 8))
sns.heatmap(pivot, cmap="coolwarm", cbar_kws={'label': 'Ratio pixels déplacés'}, linewidths=0.3, annot=False)

plt.gca().invert_yaxis()

plt.xlabel("Taux de réduction")
plt.ylabel("Nombre de fragment")

plt.tight_layout()
plt.savefig("/Users/cyriaqueperier/Documents/TIPE 2/Images/tests.jpeg", dpi=300.0)
