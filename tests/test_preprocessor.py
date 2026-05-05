"""Tests del modulo de feature engineering.

Cubre:
    * No-leakage de lags (test mas critico).
    * Stockouts implicitos: flag correcto.
    * Calendario: festivos colombianos canonicos.
    * Build feature matrix: shape, columnas y propagacion de lags al test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import (
    FEATURE_COLS,
    LAG_DAYS,
    LOCAL_TREND_WINDOW,
    TARGET_COL,
)
from src.data.loader import load_all_data
from src.data.preprocessor import (
    add_calendar_features,
    add_lag_features,
    build_feature_matrix,
    flag_implicit_stockouts,
    _propagate_lags_to_test,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_data() -> dict[str, pd.DataFrame]:
    return load_all_data()


@pytest.fixture
def synthetic_single_series() -> pd.DataFrame:
    """Una sola serie de 21 dias con valores 1..21 -- ideal para verificar lags."""
    dates = pd.date_range("2024-01-01", periods=21, freq="D")
    return pd.DataFrame(
        {
            "fecha": dates,
            "id_tienda": "STORE_TST",
            "id_producto": "PROD_TST",
            "unidades_vendidas": np.arange(1, 22, dtype=int),
        }
    )


# ---------------------------------------------------------------------------
# 1. No-leakage: test mas importante
# ---------------------------------------------------------------------------
def test_lag_no_leakage(synthetic_single_series: pd.DataFrame) -> None:
    """lag_k en la fila i debe ser exactamente sales[i - k] y NaN si i < k."""
    out = (
        add_lag_features(synthetic_single_series)
        .sort_values("fecha")
        .reset_index(drop=True)
    )
    sales = synthetic_single_series.sort_values("fecha")["unidades_vendidas"].to_numpy()

    for k in LAG_DAYS:
        col = f"lag_{k}"
        # filas 0..k-1 deben ser NaN
        assert out[col].iloc[:k].isna().all(), (
            f"{col} debe ser NaN antes de la fila {k}"
        )
        # filas k..N-1 deben coincidir con sales[i - k]
        for i in range(k, len(out)):
            assert out[col].iloc[i] == sales[i - k], (
                f"{col} en fila {i} deberia ser {sales[i - k]}, fue {out[col].iloc[i]}"
            )


def test_rolling_mean_uses_only_past(synthetic_single_series: pd.DataFrame) -> None:
    """rolling_mean_7 en fila i debe ser mean(sales[i-7..i-1]); NaN si i < 7."""
    out = (
        add_lag_features(synthetic_single_series)
        .sort_values("fecha")
        .reset_index(drop=True)
    )
    sales = synthetic_single_series.sort_values("fecha")["unidades_vendidas"].to_numpy()
    assert out["rolling_mean_7"].iloc[:7].isna().all()
    expected_at_7 = float(np.mean(sales[0:7]))  # 1..7 -> mean = 4
    assert out["rolling_mean_7"].iloc[7] == pytest.approx(expected_at_7)


def test_local_trend_slope_does_not_peek(synthetic_single_series: pd.DataFrame) -> None:
    """local_trend_slope NO debe usar el dia actual."""
    out = (
        add_lag_features(synthetic_single_series)
        .sort_values("fecha")
        .reset_index(drop=True)
    )
    # Antes de tener LOCAL_TREND_WINDOW dias historicos, slope debe ser NaN.
    assert out["local_trend_slope"].iloc[:LOCAL_TREND_WINDOW].isna().all()
    # En la primera fila valida (idx = LOCAL_TREND_WINDOW), la ventana usa
    # sales[0..LOCAL_TREND_WINDOW-1] = 1..LOCAL_TREND_WINDOW -> slope = 1.0.
    assert out["local_trend_slope"].iloc[LOCAL_TREND_WINDOW] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. Stockouts implicitos
# ---------------------------------------------------------------------------
def test_flag_implicit_stockouts_basic() -> None:
    """Un cero rodeado por positivos en la misma serie debe marcarse."""
    df = pd.DataFrame(
        {
            "fecha": pd.date_range("2024-01-01", periods=5, freq="D"),
            "id_tienda": ["A"] * 5,
            "id_producto": ["X"] * 5,
            "unidades_vendidas": [3, 0, 4, 0, 5],
        }
    )
    out = flag_implicit_stockouts(df).sort_values("fecha").reset_index(drop=True)
    # idx 1: 0 entre 3 y 4 -> stockout
    # idx 3: 0 entre 4 y 5 -> stockout
    assert list(out["is_implicit_stockout"]) == [False, True, False, True, False]


def test_flag_implicit_stockouts_no_cross_contamination() -> None:
    """Series distintas no deben contaminarse via shift."""
    df = pd.DataFrame(
        {
            "fecha": list(pd.date_range("2024-01-01", periods=2, freq="D")) * 2,
            "id_tienda": ["A", "A", "B", "B"],
            "id_producto": ["X", "X", "X", "X"],
            "unidades_vendidas": [3, 0, 0, 5],
        }
    )
    out = flag_implicit_stockouts(df)
    # Ningun cero deberia cumplir la triada porque ninguno tiene vecinos
    # positivos a ambos lados dentro de su propia serie.
    assert out["is_implicit_stockout"].sum() == 0


# ---------------------------------------------------------------------------
# 3. Calendario / festivos colombianos
# ---------------------------------------------------------------------------
def test_calendar_independence_day_is_holiday() -> None:
    """20 de julio (Dia de la Independencia CO) debe tener is_holiday=1."""
    df = pd.DataFrame(
        {
            "fecha": [pd.Timestamp("2024-07-20")],
            "id_tienda": ["A"],
            "id_producto": ["X"],
            "unidades_vendidas": [1],
        }
    )
    out = add_calendar_features(df)
    assert int(out["is_holiday"].iloc[0]) == 1


def test_calendar_normal_workday_not_holiday() -> None:
    """Un miercoles cualquiera no festivo debe tener is_holiday=0 y is_weekend=0."""
    df = pd.DataFrame(
        {
            "fecha": [pd.Timestamp("2024-02-21")],  # miercoles
            "id_tienda": ["A"],
            "id_producto": ["X"],
            "unidades_vendidas": [1],
        }
    )
    out = add_calendar_features(df)
    row = out.iloc[0]
    assert int(row["is_holiday"]) == 0
    assert int(row["is_weekend"]) == 0
    assert int(row["day_of_week"]) == 2  # miercoles


def test_calendar_friday_is_weekend_per_skill_convention() -> None:
    """is_weekend incluye Vie-Dom segun el skill (no solo Sab-Dom)."""
    df = pd.DataFrame(
        {
            "fecha": [pd.Timestamp("2024-02-23")],  # viernes
            "id_tienda": ["A"],
            "id_producto": ["X"],
            "unidades_vendidas": [1],
        }
    )
    out = add_calendar_features(df)
    assert int(out["is_weekend"].iloc[0]) == 1


def test_days_to_next_holiday_capped() -> None:
    """days_to_next_holiday se acota a HOLIDAY_HORIZON_DAYS (=30)."""
    df = pd.DataFrame(
        {
            "fecha": [
                pd.Timestamp("2024-04-15")
            ],  # ~5 dias antes del 1 de mayo (Dia del Trabajo CO)
            "id_tienda": ["A"],
            "id_producto": ["X"],
            "unidades_vendidas": [1],
        }
    )
    out = add_calendar_features(df)
    val = int(out["days_to_next_holiday"].iloc[0])
    assert 1 <= val <= 30


# ---------------------------------------------------------------------------
# 4. Pipeline orquestador
# ---------------------------------------------------------------------------
def test_build_feature_matrix_shape(real_data: dict[str, pd.DataFrame]) -> None:
    """X_train debe perder 14 dias por combo (los lags mas largos) y conservar 160 series."""
    ventas = real_data["ventas"]
    cutoff = ventas["fecha"].max() - pd.Timedelta(days=7)
    train = ventas.loc[ventas["fecha"] <= cutoff]
    test = ventas.loc[ventas["fecha"] > cutoff]

    X_train, X_test = build_feature_matrix(
        train,
        test,
        maestro_tiendas=real_data["tiendas"],
        catalogo=real_data["catalogo"],
        drop_train_stockouts=False,  # para mantener el shape predecible en este test
    )
    n_combos = train.groupby(["id_tienda", "id_producto"]).ngroups
    n_train_days = train["fecha"].nunique()
    expected_train_rows = (n_train_days - LOCAL_TREND_WINDOW) * n_combos

    assert X_train.shape[0] == expected_train_rows
    assert X_test.shape[0] == len(test)
    assert all(c in X_train.columns for c in FEATURE_COLS)
    assert all(c in X_test.columns for c in FEATURE_COLS)
    assert TARGET_COL in X_train.columns and TARGET_COL not in X_test.columns


def test_build_feature_matrix_no_nans_in_train(
    real_data: dict[str, pd.DataFrame],
) -> None:
    """Tras drop_lag_nans, X_train no debe tener NaN en columnas FEATURE_COLS."""
    ventas = real_data["ventas"]
    cutoff = ventas["fecha"].max() - pd.Timedelta(days=7)
    X_train, _ = build_feature_matrix(
        ventas.loc[ventas["fecha"] <= cutoff],
        ventas.loc[ventas["fecha"] > cutoff],
        maestro_tiendas=real_data["tiendas"],
        catalogo=real_data["catalogo"],
    )
    nans = X_train[list(FEATURE_COLS)].isna().sum()
    assert nans.sum() == 0, f"NaN encontrados en X_train: {nans[nans > 0].to_dict()}"


def test_propagate_lags_first_test_day_uses_last_train_day(
    real_data: dict[str, pd.DataFrame],
) -> None:
    """En la primera fecha de test, lag_1 debe igualar el target del ultimo dia del train."""
    ventas = real_data["ventas"]
    cutoff = pd.Timestamp("2024-03-20")
    train = ventas.loc[ventas["fecha"] <= cutoff].copy()
    test = ventas.loc[ventas["fecha"] > cutoff].copy()

    test_with_lags = _propagate_lags_to_test(train, test)
    first_test_date = test_with_lags["fecha"].min()
    sample = test_with_lags.loc[test_with_lags["fecha"] == first_test_date]
    train_last_day = train.loc[train["fecha"] == cutoff].set_index(
        ["id_tienda", "id_producto"]
    )[TARGET_COL]

    for _, row in sample.iterrows():
        key = (row["id_tienda"], row["id_producto"])
        assert row["lag_1"] == pytest.approx(float(train_last_day.loc[key]))


def test_categorical_feature_dtypes_are_int(real_data: dict[str, pd.DataFrame]) -> None:
    """Las features categoricas deben quedar como enteros (no float, no string)."""
    ventas = real_data["ventas"]
    cutoff = ventas["fecha"].max() - pd.Timedelta(days=7)
    X_train, _ = build_feature_matrix(
        ventas.loc[ventas["fecha"] <= cutoff],
        ventas.loc[ventas["fecha"] > cutoff],
        maestro_tiendas=real_data["tiendas"],
        catalogo=real_data["catalogo"],
    )
    for col in [
        "day_of_week",
        "is_weekend",
        "is_holiday",
        "ciudad_encoded",
        "store_tier_encoded",
        "categoria_encoded",
        "id_tienda_encoded",
        "id_producto_encoded",
    ]:
        assert pd.api.types.is_integer_dtype(X_train[col]), (
            f"{col} deberia ser integer, fue {X_train[col].dtype}"
        )
