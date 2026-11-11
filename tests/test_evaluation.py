import numpy as np
import pandas as pd
import pytest

from ml.src.evaluation import compute_metrics, engine_folds, evaluate_on_zone
from ml.src.models import MODEL_NAMES, create_model, cross_validate, train_and_predict


# --- Métriques ------------------------------------------------------------------

def test_metrics_on_small_example():
    """RUL réels 10, 50, 100 ; prédits 12, 40, 110 : erreurs 2, -10, 10."""
    metrics = compute_metrics([10, 50, 100], [12, 40, 110])
    assert metrics["MAE"] == pytest.approx(22 / 3)
    assert metrics["RMSE"] == pytest.approx(np.sqrt(204 / 3))
    assert metrics["R2"] == pytest.approx(1 - 204 / np.sum((np.array([10, 50, 100]) - 160 / 3) ** 2))


def test_perfect_prediction():
    assert compute_metrics([1, 2, 3], [1, 2, 3]) == {"MAE": 0, "RMSE": 0, "R2": 1}


def test_predicting_the_mean_gives_r2_zero():
    assert compute_metrics([0, 10, 20], [10, 10, 10])["R2"] == pytest.approx(0)


def test_rmse_is_never_smaller_than_mae():
    metrics = compute_metrics([0, 0, 0, 0], [1, 1, 1, 9])
    assert metrics["RMSE"] >= metrics["MAE"]


def test_zone_uses_true_rul_only():
    """Seules les lignes avec un RUL réel <= horizon comptent, et on compare au vrai RUL."""
    assert evaluate_on_zone(pd.Series([0, 50, 127, 300]), [0, 50, 127, 110])["RMSE"] == 0


def test_zone_ignores_predictions_outside_zone():
    true_rul = [10, 20, 200]
    assert evaluate_on_zone(true_rul, [12, 18, 0]) == evaluate_on_zone(true_rul, [12, 18, 999])


def test_zone_without_rows_raises():
    with pytest.raises(ValueError):
        evaluate_on_zone([200, 300], [1, 2], horizon=127)


# --- Validation croisée par moteur ----------------------------------------------------

def test_engine_folds_keep_engines_together():
    df = pd.DataFrame({"unit_id": np.repeat(np.arange(1, 11), 5)})       # 10 moteurs, 5 lignes chacun
    folds = engine_folds(df, n_splits=5)
    assert len(folds) == 5
    controlled = []
    for fit_engines, check_engines in folds:
        assert set(fit_engines).isdisjoint(check_engines)
        assert sorted(fit_engines + check_engines) == list(range(1, 11))
        controlled += check_engines
    assert sorted(controlled) == list(range(1, 11))                       # chaque moteur contrôlé une fois


def test_engine_folds_are_deterministic():
    df = pd.DataFrame({"unit_id": np.repeat(np.arange(1, 21), 3)})
    assert engine_folds(df) == engine_folds(df)


# --- Modèles ------------------------------------------------------------------

def test_create_model():
    assert type(create_model("Régression linéaire")).__name__ == "LinearRegression"
    assert create_model("Ridge", alpha=10).named_steps["ridge"].alpha == 10
    assert create_model("Forêt aléatoire", max_features=1.0).max_features == 1.0
    assert create_model("Forêt aléatoire").max_features == 0.33
    assert len(MODEL_NAMES) == 3
    with pytest.raises(ValueError):
        create_model("inconnu")


@pytest.fixture
def linear_engines():
    """6 moteurs où la feature x vaut exactement le RUL : un modèle linéaire doit le retrouver."""
    rows = []
    for unit_id in range(1, 7):
        for rul in range(20, -1, -1):
            rows.append({"unit_id": unit_id, "x": float(rul), "RUL": rul})
    return pd.DataFrame(rows)


def test_train_and_predict_cap_and_no_negative(linear_engines):
    """Le plafond ne change que la cible d'apprentissage, et aucune prédiction n'est négative."""
    model, predictions = train_and_predict("Forêt aléatoire", linear_engines, linear_engines, ["x"], cap=10)
    assert predictions.max() <= 10 + 1e-9
    shifted = linear_engines.assign(x=linear_engines["x"] - 30)            # extrapolation vers des RUL < 0
    _, predictions = train_and_predict("Régression linéaire", linear_engines, shifted, ["x"])
    assert predictions.min() == 0


def test_cross_validate_on_easy_problem(linear_engines):
    folds = engine_folds(linear_engines, n_splits=3)
    for model_name in ["Régression linéaire", "Ridge"]:
        result = cross_validate(model_name, linear_engines, folds, ["x"], alpha=1e-6)
        assert result["RMSE"] == pytest.approx(0, abs=1e-3)
