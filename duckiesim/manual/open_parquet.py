#!/usr/bin/env python
# analyze_parquet.py

"""
This script opens a Parquet file located in the 'dataset' directory and outputs statistics:
- Number of columns
- Column names
- Data dimensions
- Shape of elements in complex columns (e.g., arrays)
"""

import os
import pandas as pd
import numpy as np

# --- Arguments de la ligne de commande ---
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file-name", 
                    default="expert_data_64_.parquet",
                    help="Name of the Parquet file to analyze")
args = parser.parse_args()

# --- Configuration du chemin du fichier Parquet ---
script_dir = os.path.dirname(os.path.realpath(__file__))  # Répertoire du script
input_file = os.path.join(script_dir, "dataset", args.file_name)  # Chemin du fichier dans 'dataset'

# --- Vérification de l'existence du fichier ---
if not os.path.exists(input_file):
    print(f"Erreur : Le fichier '{input_file}' n'existe pas.")
    exit(1)

# --- Chargement des données ---
df = pd.read_parquet(input_file, engine="pyarrow")

# --- Conversion des données en numpy ---
s_data = np.array(df["s"].tolist())
a_data = np.array(df["a"].tolist()) if "a" in df.columns else None
r_data = df["r"].to_numpy()
d_data = df["d"].to_numpy()
next_s_data = np.array(df["next_s"].tolist())

# --- Analyse et statistiques ---
print("\n--- Analyse du fichier Parquet ---\n")

# Nombre de colonnes et noms des colonnes
print(f"Nombre de colonnes : {df.shape[1]}")
print(f"Noms des colonnes : {list(df.columns)}")

# Dimensions des données
print("\n--- Dimensions des datasets numpy ---")
print(f"s_data shape      : {s_data.shape}")
print(f"a_data shape      : {a_data.shape if a_data is not None else 'Non disponible'}")
print(f"r_data shape      : {r_data.shape}")
print(f"d_data shape      : {d_data.shape}")
print(f"next_s_data shape : {next_s_data.shape}")

# --- Statistiques détaillées ---
print("\n--- Statistiques des colonnes ---")

# Statistiques pour `s_data`
print("\nColonne 's'")
print(f" - Type : {s_data.dtype}")
print(f" - Dimension par échantillon : {s_data[0].shape if len(s_data) > 0 else 'N/A'}")
print(f" - Nombre total d'échantillons : {s_data.shape[0]}")

# Statistiques pour `a_data`
if a_data is not None:
    print("\nColonne 'a'")
    print(f" - Type : {a_data.dtype}")
    print(f" - Dimension par échantillon : {a_data[0].shape if len(a_data) > 0 else 'N/A'}")
    print(f" - Nombre total d'échantillons : {a_data.shape[0]}")

# Statistiques pour `r_data`
print("\nColonne 'r'")
print(f" - Type : {r_data.dtype}")
print(f" - Min : {r_data.min()}")
print(f" - Max : {r_data.max()}")
print(f" - Moyenne : {r_data.mean()}")
print(f" - Écart-type : {r_data.std()}")

# Statistiques pour `d_data`
print("\nColonne 'd'")
print(f" - Type : {d_data.dtype}")
print(f" - Nombre de vrais : {np.sum(d_data)}")
print(f" - Nombre de faux : {len(d_data) - np.sum(d_data)}")

# Statistiques pour `next_s_data`
print("\nColonne 'next_s'")
print(f" - Type : {next_s_data.dtype}")
print(f" - Dimension par échantillon : {next_s_data[0].shape if len(next_s_data) > 0 else 'N/A'}")
print(f" - Nombre total d'échantillons : {next_s_data.shape[0]}")
