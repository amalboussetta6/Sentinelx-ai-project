"""Métriques et validation croisée par moteur."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


def compute_metrics(y_true, y_pred):
    """MAE et RMSE (en cycles) et R²."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluate_on_zone(true_rul, predictions, horizon=127):
    """Métriques sur la zone utile (RUL réel <= horizon), toujours comparées au vrai RUL (jamais plafonné)."""
    true_rul = np.asarray(true_rul)
    predictions = np.asarray(predictions)
    in_zone = true_rul <= horizon
    if not in_zone.any():
        raise ValueError("Aucune ligne dans la zone d'évaluation.")
    return compute_metrics(true_rul[in_zone], predictions[in_zone])


def engine_folds(df, n_splits=5):
    """Tours de validation croisée par moteur (GroupKFold) : liste de (moteurs pour apprendre, moteurs contrôlés)."""
    folds = []
    for fit_index, check_index in GroupKFold(n_splits=n_splits).split(df, groups=df["unit_id"]):
        fit_engines = sorted(int(e) for e in df["unit_id"].iloc[fit_index].unique())
        check_engines = sorted(int(e) for e in df["unit_id"].iloc[check_index].unique())
        folds.append((fit_engines, check_engines))
    return folds
