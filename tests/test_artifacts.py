"""Cohérence entre données réelles, artefacts des notebooks et ml/src (ignorés si les fichiers sont absents)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ml.src.data import COLUMN_NAMES, add_rul, last_cycle_per_engine, load_cmapss, split_engines
from ml.src.evaluation import compute_metrics, engine_folds
from ml.src.features import add_history_features, history_feature_names

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EDA = ROOT / "artifacts" / "eda_fd001"
BASELINE = ROOT / "artifacts" / "baseline_fd001"

needs_data = pytest.mark.skipif(not (RAW / "train_FD001.txt").exists(), reason="données FD001 absentes de data/raw")
needs_eda = pytest.mark.skipif(not (EDA / "fd001_engine_split.json").exists(), reason="lancer d'abord le notebook EDA")
needs_baseline = pytest.mark.skipif(not (BASELINE / "baseline_rul_fd001.joblib").exists(),
                                    reason="lancer d'abord le notebook Baseline")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@needs_data
def test_real_files_are_consistent():
    train = add_rul(load_cmapss(RAW / "train_FD001.txt"))
    test = load_cmapss(RAW / "test_FD001.txt")
    assert list(test.columns) == COLUMN_NAMES and len(train) == 20631 and len(test) == 13096
    assert train["unit_id"].nunique() == 100 == test["unit_id"].nunique()
    assert train.isna().sum().sum() == 0 and not train.duplicated(["unit_id", "cycle"]).any()
    assert (train.groupby("unit_id")["cycle"].diff().dropna() == 1).all()
    assert (train.groupby("unit_id")["RUL"].min() == 0).all()
    assert last_cycle_per_engine(test)["cycle"].min() == 31
    assert len(np.loadtxt(RAW / "RUL_FD001.txt")) == 100


@needs_data
@needs_eda
def test_eda_split_is_reproducible_and_by_engine():
    split = read_json(EDA / "fd001_engine_split.json")
    train_engines, validation_engines = split_engines(range(1, 101), 20, split["random_state"])
    assert split["train_engines"] == train_engines and split["validation_engines"] == validation_engines
    assert set(train_engines).isdisjoint(validation_engines)


@needs_data
@needs_eda
def test_selected_sensors_vary_and_excluded_do_not():
    groups = read_json(EDA / "fd001_feature_groups.json")
    train = load_cmapss(RAW / "train_FD001.txt")
    assert len(groups["selected_sensors"]) == 14
    assert (train[groups["selected_sensors"]].nunique() > 2).all()
    assert (train[groups["constant_columns"]].nunique() == 1).all()
    assert not set(groups["selected_sensors"]) & set(groups["forbidden_as_feature"] + groups["excluded_columns"])


@needs_data
@needs_baseline
def test_baseline_reuses_eda_decisions_and_folds():
    metrics = read_json(BASELINE / "baseline_metrics.json")
    split = read_json(EDA / "fd001_engine_split.json")
    assert metrics["features"] == history_feature_names(metrics["sensors"])
    assert metrics["validation_engines"] == split["validation_engines"]
    train = load_cmapss(RAW / "train_FD001.txt")
    train_part = train[train["unit_id"].isin(split["train_engines"])]
    assert [check for _, check in engine_folds(train_part)] == metrics["cross_validation"]["fold_engines"]
    assert not {e for fold in metrics["cross_validation"]["fold_engines"] for e in fold} & set(split["validation_engines"])


@needs_data
@needs_baseline
def test_saved_model_reproduces_saved_test_predictions():
    metrics = read_json(BASELINE / "baseline_metrics.json")
    bundle = joblib.load(BASELINE / "baseline_rul_fd001.joblib")
    assert bundle["model_name"] == metrics["selected_model"] and bundle["features"] == metrics["features"]
    test = add_history_features(load_cmapss(RAW / "test_FD001.txt"), bundle["sensors"], bundle["window"], bundle["n_start"])
    predictions = np.clip(bundle["model"].predict(last_cycle_per_engine(test)[bundle["features"]]), 0, None)
    saved = pd.read_csv(BASELINE / "test_predictions.csv")
    np.testing.assert_allclose(predictions, saved[bundle["model_name"]], atol=1e-3)
    for model_name, values in metrics["test_metrics_last_cycle"].items():
        assert compute_metrics(saved["RUL_true"], saved[model_name])["RMSE"] == pytest.approx(values["RMSE"], abs=1e-3)
