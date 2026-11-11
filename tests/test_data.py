import pandas as pd
import pytest

from ml.src.data import COLUMN_NAMES, add_rul, last_cycle_per_engine, load_cmapss, split_engines


def test_rul_counts_down_to_zero(two_engines):
    df = add_rul(two_engines)
    assert df[df["unit_id"] == 1]["RUL"].tolist() == list(range(11, -1, -1))   # 11, 10, ..., 0
    assert df[df["unit_id"] == 2]["RUL"].tolist() == list(range(7, -1, -1))
    assert (df["RUL"] >= 0).all()


def test_rul_is_computed_per_engine_even_if_rows_are_shuffled(two_engines):
    shuffled = two_engines.sample(frac=1, random_state=0)
    df = add_rul(shuffled)
    assert (df["RUL"] == df.groupby("unit_id")["cycle"].transform("max") - df["cycle"]).all()


def test_rul_does_not_change_input(two_engines):
    before = two_engines.copy()
    add_rul(two_engines)
    assert two_engines.equals(before)


def test_single_cycle_engine_has_rul_zero():
    df = add_rul(pd.DataFrame({"unit_id": [7], "cycle": [1]}))
    assert df["RUL"].tolist() == [0]


def test_last_cycle_per_engine(two_engines):
    last = last_cycle_per_engine(two_engines.iloc[::-1])          # ordre inversé en entrée
    assert last["unit_id"].tolist() == [1, 2] and last["cycle"].tolist() == [12, 8]


def test_load_cmapss_names_and_sorts_columns(tmp_path):
    line_a = " ".join(["2", "1"] + ["0.5"] * 24)
    line_b = " ".join(["1", "1"] + ["0.5"] * 24)
    path = tmp_path / "mini.txt"
    path.write_text(line_a + "  \n" + line_b + "\n")                # espaces en fin de ligne, comme les fichiers NASA
    df = load_cmapss(path)
    assert list(df.columns) == COLUMN_NAMES and len(COLUMN_NAMES) == 26
    assert df["unit_id"].tolist() == [1, 2]
    assert df.isna().sum().sum() == 0


def test_split_has_no_engine_overlap_and_keeps_all_engines():
    train, validation = split_engines(range(1, 101), n_validation=20, random_state=42)
    assert len(train) == 80 and len(validation) == 20
    assert set(train).isdisjoint(validation)
    assert sorted(train + validation) == list(range(1, 101))


def test_split_is_reproducible_and_depends_on_seed():
    assert split_engines(range(1, 101), 20, 42) == split_engines(range(1, 101), 20, 42)
    assert split_engines(range(1, 101), 20, 42) != split_engines(range(1, 101), 20, 0)


def test_split_ignores_duplicated_engine_ids():
    """On passe une colonne unit_id (une valeur par ligne) : chaque moteur ne compte qu'une fois."""
    train, validation = split_engines([1, 1, 1, 2, 2, 3, 3, 4], n_validation=1)
    assert len(validation) == 1 and len(train) == 3


@pytest.mark.parametrize("rows_engine_1", [1, 5])
def test_rul_zero_exactly_once_per_engine(rows_engine_1):
    df = add_rul(pd.DataFrame({"unit_id": [1] * rows_engine_1 + [2, 2], "cycle": list(range(1, rows_engine_1 + 1)) + [1, 2]}))
    assert (df.groupby("unit_id")["RUL"].apply(lambda r: (r == 0).sum()) == 1).all()
