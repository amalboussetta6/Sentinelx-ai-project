"""Features d'historique calculées moteur par moteur, avec le passé uniquement (calcul causal)."""
import numpy as np
import pandas as pd


def rolling_slope(values, window):
    """Pour chaque cycle : pente (unités par cycle) de la droite ajustée sur les `window` dernières valeurs.

    `values` contient les mesures d'un seul moteur, dans l'ordre des cycles. Pente = 0 au premier cycle.
    """
    values = np.asarray(values, dtype=float)
    slopes = []
    for position in range(len(values)):
        recent = values[max(0, position - window + 1):position + 1]   # au plus `window` valeurs déjà observées
        if len(recent) < 2:
            slopes.append(0.0)
        else:
            slope, _ = np.polyfit(np.arange(len(recent)), recent, 1)
            slopes.append(slope)
    return np.array(slopes)


def delta_from_start(values, cycles, n_start):
    """Mesure actuelle - niveau de départ du moteur.

    Niveau de départ = moyenne des `n_start` premiers cycles ; avant le cycle `n_start`,
    moyenne des cycles déjà vus (on n'utilise jamais un cycle futur).
    """
    reference = values.expanding().mean()
    reference[cycles > n_start] = values[cycles <= n_start].mean()
    return values - reference


def add_history_features(df, sensors, window=30, n_start=20):
    """Ajoute <capteur>_mean, <capteur>_slope et <capteur>_delta, en traitant les moteurs un par un."""
    engine_tables = []
    for _, engine in df.sort_values(["unit_id", "cycle"]).groupby("unit_id"):
        engine = engine.copy()
        for sensor in sensors:
            values = engine[sensor]
            engine[sensor + "_mean"] = values.rolling(window, min_periods=1).mean()
            engine[sensor + "_slope"] = rolling_slope(values, window)
            engine[sensor + "_delta"] = delta_from_start(values, engine["cycle"], n_start)
        engine_tables.append(engine)
    return pd.concat(engine_tables)


def history_feature_names(sensors):
    """Noms des 3 x len(sensors) features : toutes les _mean, puis les _slope, puis les _delta."""
    return [sensor + suffix for suffix in ["_mean", "_slope", "_delta"] for sensor in sensors]
