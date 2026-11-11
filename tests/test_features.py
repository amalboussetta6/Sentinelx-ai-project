import numpy as np
import pandas as pd
import pytest

from ml.src.features import add_history_features, delta_from_start, history_feature_names, rolling_slope

FEATURES = history_feature_names(["sensor_2"])


def test_slope_recovers_known_trend(two_engines):
    df = add_history_features(two_engines, ["sensor_2"], window=4, n_start=3)
    engine_1 = df[df["unit_id"] == 1]["sensor_2_slope"].to_numpy()
    engine_2 = df[df["unit_id"] == 2]["sensor_2_slope"].to_numpy()
    assert engine_1[0] == 0                                     # un seul cycle : pas de pente
    np.testing.assert_allclose(engine_1[1:], 0.5)
    np.testing.assert_allclose(engine_2[1:], -2.0)


def test_rolling_slope_first_cycle_and_window():
    slopes = rolling_slope([0.0, 10.0, 11.0, 12.0], window=3)
    assert slopes[0] == 0
    assert slopes[1] == pytest.approx(10.0)                     # fenêtre [0, 10]
    assert slopes[3] == pytest.approx(1.0)                      # fenêtre [10, 11, 12] : le 0 est sorti


def test_constant_signal_has_zero_slope_and_zero_delta():
    toy = pd.DataFrame({"unit_id": 1, "cycle": range(1, 8), "s": 5.0})
    df = add_history_features(toy, ["s"], window=3, n_start=2)
    np.testing.assert_allclose(df["s_slope"], 0, atol=1e-12)
    np.testing.assert_allclose(df["s_delta"], 0)
    np.testing.assert_allclose(df["s_mean"], 5.0)


def test_delta_reference_before_and_after_start():
    """Avant n_start : moyenne des cycles déjà vus ; ensuite : moyenne figée des n_start premiers cycles."""
    values = pd.Series([100.0, 102.0, 101.0, 105.0])
    cycles = pd.Series([1, 2, 3, 4])
    np.testing.assert_allclose(delta_from_start(values, cycles, n_start=2), [0, 1, 0, 4])


def test_engine_shorter_than_n_start():
    """Un moteur plus court que le début de vie : seule la moyenne des cycles vus sert de référence."""
    values = pd.Series([10.0, 20.0])
    np.testing.assert_allclose(delta_from_start(values, pd.Series([1, 2]), n_start=20), [0, 5])


def test_fictional_engine_computed_by_hand():
    """Exemple de 5 cycles du notebook de baseline, calculé à la main (fenêtre 3, début de vie 2)."""
    toy = pd.DataFrame({"unit_id": 1, "cycle": [1, 2, 3, 4, 5], "sensor_x": [100, 102, 101, 105, 108]})
    df = add_history_features(toy, ["sensor_x"], window=3, n_start=2)
    np.testing.assert_allclose(df["sensor_x_mean"], [100, 101, 101, 308 / 3, 314 / 3])
    np.testing.assert_allclose(df["sensor_x_slope"], [0, 2, 0.5, 1.5, 3.5])
    np.testing.assert_allclose(df["sensor_x_delta"], [0, 1, 0, 4, 7])


def test_future_does_not_change_past(two_engines):
    """Modifier les cycles futurs ne doit pas changer les features passées (pas de fuite du futur)."""
    before = add_history_features(two_engines, ["sensor_2"], window=4, n_start=3)
    modified = two_engines.copy()
    modified.loc[modified["cycle"] > 5, "sensor_2"] += 1000
    after = add_history_features(modified, ["sensor_2"], window=4, n_start=3)
    past = before["cycle"] <= 5
    np.testing.assert_allclose(before.loc[past, FEATURES], after.loc[past, FEATURES])


def test_truncated_engine_gives_same_features():
    """Calculer les features sur un historique coupé au cycle t donne les mêmes valeurs jusqu'à t :
    c'est exactement la situation du jeu de test NASA (moteurs arrêtés avant la panne)."""
    rng = np.random.RandomState(0)
    full = pd.DataFrame({"unit_id": 1, "cycle": range(1, 61), "s": rng.normal(size=60).cumsum()})
    on_full = add_history_features(full, ["s"], window=30, n_start=20)
    on_cut = add_history_features(full[full["cycle"] <= 25], ["s"], window=30, n_start=20)
    names = history_feature_names(["s"])
    np.testing.assert_allclose(on_full[names].iloc[:25].to_numpy(), on_cut[names].to_numpy())


def test_engines_are_not_mixed(two_engines):
    """Les features du moteur 2 ne dépendent pas du moteur 1."""
    before = add_history_features(two_engines, ["sensor_2"], window=4, n_start=3)
    modified = two_engines.copy()
    modified.loc[modified["unit_id"] == 1, "sensor_2"] *= 50
    after = add_history_features(modified, ["sensor_2"], window=4, n_start=3)
    engine_2 = before["unit_id"] == 2
    np.testing.assert_allclose(before.loc[engine_2, FEATURES], after.loc[engine_2, FEATURES])


def test_input_is_not_modified(two_engines):
    before = two_engines.copy()
    add_history_features(two_engines, ["sensor_2"])
    assert two_engines.equals(before)


def test_feature_names_order_and_no_forbidden_variable():
    names = history_feature_names(["sensor_2", "sensor_3"])
    assert names == ["sensor_2_mean", "sensor_3_mean", "sensor_2_slope", "sensor_3_slope",
                     "sensor_2_delta", "sensor_3_delta"]
    for forbidden in ["unit_id", "RUL", "cycle", "last_cycle", "life_pct"]:
        assert forbidden not in names
