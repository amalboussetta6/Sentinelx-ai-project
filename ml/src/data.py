"""Chargement des fichiers NASA C-MAPSS, cible RUL et séparation des moteurs."""
import numpy as np
import pandas as pd

COLUMN_NAMES = (["unit_id", "cycle", "setting_1", "setting_2", "setting_3"]
                + [f"sensor_{i}" for i in range(1, 22)])


def load_cmapss(path):
    """Charge un fichier C-MAPSS (valeurs séparées par des espaces, sans en-tête), trié par moteur puis cycle."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    return df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)


def add_rul(df):
    """Ajoute RUL = dernier cycle du moteur - cycle courant (moteurs suivis jusqu'à la panne)."""
    df = df.copy()
    df["RUL"] = df.groupby("unit_id")["cycle"].transform("max") - df["cycle"]
    return df


def last_cycle_per_engine(df):
    """Une ligne par moteur : son dernier cycle observé (protocole du test NASA)."""
    return df.sort_values(["unit_id", "cycle"]).groupby("unit_id").tail(1).reset_index(drop=True)


def split_engines(engine_ids, n_validation, random_state=42):
    """Sépare des moteurs (et non des lignes) : renvoie (moteurs d'entraînement, moteurs de validation), triés.

    Les moteurs sont mélangés avec une graine fixe, puis les n_validation premiers vont en validation.
    """
    shuffled = np.random.RandomState(random_state).permutation(np.sort(np.unique(engine_ids)))
    validation = sorted(int(e) for e in shuffled[:n_validation])
    train = sorted(int(e) for e in shuffled[n_validation:])
    return train, validation
