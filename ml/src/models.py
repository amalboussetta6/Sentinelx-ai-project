"""Les trois modèles de la baseline, leur entraînement et la validation croisée par moteur."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.src.evaluation import evaluate_on_zone

MODEL_NAMES = ["Régression linéaire", "Ridge", "Forêt aléatoire"]


def create_model(model_name, max_features=0.33, alpha=1.0, random_state=42):
    """Crée un modèle neuf. `max_features` concerne la forêt, `alpha` la régression Ridge.

    Ridge est précédée d'une normalisation : la pénalité doit traiter toutes les features à la même échelle.
    Dans un pipeline, la normalisation est apprise uniquement sur les données passées à fit().
    """
    if model_name == "Régression linéaire":
        return LinearRegression()
    if model_name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    if model_name == "Forêt aléatoire":
        return RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features=max_features,
                                     random_state=random_state, n_jobs=-1)
    raise ValueError(f"Modèle inconnu : {model_name}")


def train_and_predict(model_name, train_data, predict_data, feature_columns, cap=None, **params):
    """Entraîne un modèle neuf sur train_data (cible plafonnée à `cap` si demandé) et prédit predict_data.

    Les prédictions négatives sont ramenées à 0 : un RUL ne peut pas être négatif.
    """
    target = train_data["RUL"] if cap is None else train_data["RUL"].clip(upper=cap)
    model = create_model(model_name, **params)
    model.fit(train_data[feature_columns], target)
    predictions = np.clip(model.predict(predict_data[feature_columns]), 0, None)
    return model, predictions


def cross_validate(model_name, df, folds, feature_columns, cap=None, horizon=127, **params):
    """Un modèle par tour (apprend sur les moteurs `fit`, contrôlé sur les moteurs `check`), puis moyenne."""
    fold_metrics = []
    for fit_engines, check_engines in folds:
        fit_data = df[df["unit_id"].isin(fit_engines)]
        check_data = df[df["unit_id"].isin(check_engines)]
        _, predictions = train_and_predict(model_name, fit_data, check_data, feature_columns, cap, **params)
        fold_metrics.append(evaluate_on_zone(check_data["RUL"], predictions, horizon))
    table = pd.DataFrame(fold_metrics)
    return {"RMSE": table["RMSE"].mean(), "RMSE écart-type": table["RMSE"].std(),
            "MAE": table["MAE"].mean(), "R2": table["R2"].mean()}
