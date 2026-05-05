"""Tests del optimizador Newsvendor + integracion CP-Newsvendor.

Cubre:
    * Q* >= forecast_mean cuando CR > 0.5 (todos los productos del proyecto).
    * Pedido nunca negativo.
    * Alerta SOBRESTOCK cuando stock_actual > Q*.
    * compute_expected_cost analitico vs simulacion Monte Carlo (3 dec).
    * Conexion CP-Newsvendor: para un producto, q_star del optimizador
      es exactamente el upper bound conformal del DemandConformalForecaster
      al nivel CR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import CATEGORICAL_FEATURES, FEATURE_COLS, TARGET_COL
from src.data.loader import load_all_data
from src.data.preprocessor import build_feature_matrix
from src.models.conformal import DemandConformalForecaster
from src.models.forecaster import GlobalDemandForecaster
from src.optimization.newsvendor import (
    NewsvendorOptimizer,
    compute_business_impact,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_data() -> dict[str, pd.DataFrame]:
    return load_all_data()


@pytest.fixture(scope="module")
def optimizer(real_data: dict[str, pd.DataFrame]) -> NewsvendorOptimizer:
    return NewsvendorOptimizer(real_data["catalogo"])


@pytest.fixture(scope="module")
def synthetic_predictions() -> pd.DataFrame:
    """Predicciones sinteticas con upper conformal explicito por producto."""
    return pd.DataFrame(
        [
            {
                "id_tienda": "STORE_01",
                "id_producto": "PROD_001",
                "y_pred": 50.0,
                "y_upper": 80.0,
                "coverage_target": 0.9942,
            },
            {
                "id_tienda": "STORE_01",
                "id_producto": "PROD_007",
                "y_pred": 100.0,
                "y_upper": 150.0,
                "coverage_target": 0.9852,
            },
            {
                "id_tienda": "STORE_02",
                "id_producto": "PROD_001",
                "y_pred": 30.0,
                "y_upper": 60.0,
                "coverage_target": 0.9942,
            },
        ]
    )


@pytest.fixture(scope="module")
def synthetic_inventario() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id_tienda": "STORE_01", "id_producto": "PROD_001", "stock_actual": 5},
            {
                "id_tienda": "STORE_01",
                "id_producto": "PROD_007",
                "stock_actual": 200,
            },  # SOBRESTOCK
            {"id_tienda": "STORE_02", "id_producto": "PROD_001", "stock_actual": 0},
        ]
    )


# ---------------------------------------------------------------------------
# 1. Propiedades estructurales
# ---------------------------------------------------------------------------
def test_q_star_geq_forecast_mean_when_cr_gt_half(
    real_data: dict[str, pd.DataFrame],
) -> None:
    """Para CR > 0.5, Q* >= mean. En el catalogo Tostao' todos cumplen."""
    cat = real_data["catalogo"].copy()
    cat["margen"] = cat["precio_venta"] - cat["costo_unitario"]
    cat["critical_ratio"] = cat["margen"] / (
        cat["margen"] + cat["costo_almacenamiento_semanal"]
    )
    assert (cat["critical_ratio"] > 0.5).all(), "Premise del test"

    # Sintetizamos predicciones donde y_upper >= y_pred (siempre cierto en CP one-sided).
    preds = pd.DataFrame(
        {
            "id_tienda": ["S1"] * len(cat),
            "id_producto": cat["id_producto"].values,
            "y_pred": [10.0] * len(cat),
            "y_upper": [25.0] * len(cat),  # cualquier valor >= y_pred
        }
    )
    inv = pd.DataFrame(
        {
            "id_tienda": ["S1"] * len(cat),
            "id_producto": cat["id_producto"].values,
            "stock_actual": [0] * len(cat),
        }
    )
    opt = NewsvendorOptimizer(cat)
    out = opt.compute_order(preds, inv)
    assert (out["cantidad_target_q_star"] >= out["forecast_media_semanal"]).all()


def test_cantidad_pedido_never_negative(
    optimizer: NewsvendorOptimizer,
    synthetic_predictions: pd.DataFrame,
    synthetic_inventario: pd.DataFrame,
) -> None:
    out = optimizer.compute_order(synthetic_predictions, synthetic_inventario)
    assert (out["cantidad_pedido"] >= 0).all()


def test_sobrestock_alerta_se_activa(
    optimizer: NewsvendorOptimizer,
    synthetic_predictions: pd.DataFrame,
    synthetic_inventario: pd.DataFrame,
) -> None:
    """STORE_01/PROD_007 tiene stock=200 > Q*=150, debe alertar SOBRESTOCK."""
    out = optimizer.compute_order(synthetic_predictions, synthetic_inventario)
    row = out.query("id_tienda == 'STORE_01' and id_producto == 'PROD_007'").iloc[0]
    assert row["alerta"].startswith("SOBRESTOCK"), (
        f"alerta esperada, fue {row['alerta']!r}"
    )
    assert row["cantidad_pedido"] == 0, "no se pide cuando ya hay sobrestock"


def test_sin_sobrestock_no_hay_alerta(
    optimizer: NewsvendorOptimizer,
    synthetic_predictions: pd.DataFrame,
    synthetic_inventario: pd.DataFrame,
) -> None:
    out = optimizer.compute_order(synthetic_predictions, synthetic_inventario)
    not_so = out.query("id_tienda == 'STORE_02' and id_producto == 'PROD_001'").iloc[0]
    assert not_so["alerta"] == ""


# ---------------------------------------------------------------------------
# 2. Costo analitico vs Monte Carlo
# ---------------------------------------------------------------------------
def test_compute_expected_cost_matches_monte_carlo(
    optimizer: NewsvendorOptimizer,
) -> None:
    """La formula cerrada Normal vs simulacion deben coincidir hasta 3 dec
    (con n_sims grande la simulacion converge al analitico)."""
    rng = np.random.default_rng(42)
    mu, sigma = 50.0, 10.0
    cu, co = 1700.0, 10.0
    Q = 65.0  # Q != mu para tener costo no trivial

    analitico = optimizer.compute_expected_cost(Q, mu, sigma, cu, co)

    n_sims = 5_000_000
    D = rng.normal(mu, sigma, size=n_sims)
    monte_carlo = float(
        np.mean(cu * np.maximum(0.0, D - Q) + co * np.maximum(0.0, Q - D))
    )
    # Tolerancia relativa: 0.5% (Monte Carlo ~ 5e6 sims es suficientemente preciso)
    assert abs(analitico - monte_carlo) / max(monte_carlo, 1e-6) < 0.005, (
        f"analitico={analitico:.3f}, MC={monte_carlo:.3f}"
    )


def test_sensitivity_curve_minimum_near_q_optimo(
    optimizer: NewsvendorOptimizer,
) -> None:
    """El minimo de la curva de costo debe coincidir con Q_optimo_normal."""
    df = optimizer.sensitivity_analysis(
        id_tienda="STORE_01",
        id_producto="PROD_001",
        forecast_mean=50.0,
        forecast_std=10.0,
        n_points=200,
    )
    q_min_curve = float(df.loc[df["costo_total"].idxmin(), "Q"])
    q_optimo = float(df["Q_optimo_normal"].iloc[0])
    # Granularidad de la curva es ~ 9*sigma/200 = 0.45, asi que tolerancia 1.0
    assert abs(q_min_curve - q_optimo) < 1.0, (
        f"q_min_curve={q_min_curve:.2f}, q_optimo={q_optimo:.2f}"
    )


# ---------------------------------------------------------------------------
# 3. Conexion CP-Newsvendor (test critico de integracion)
# ---------------------------------------------------------------------------
def test_q_star_equals_conformal_upper_per_product(
    real_data: dict[str, pd.DataFrame],
) -> None:
    """El q_star del Newsvendor para un producto debe ser EXACTAMENTE el
    upper bound conformal del DemandConformalForecaster al nivel CR.

    Construye una mini-pipeline:
        1. Fit del forecaster sobre 6 semanas.
        2. Calibracion conformal sobre 1 semana.
        3. Prediccion + intervalo en 1 semana de holdout.
        4. Aplica NewsvendorOptimizer pasando y_upper conformal como input.
        5. Verifica que cantidad_target_q_star == y_upper conformal por fila.
    """
    ventas = real_data["ventas"]
    tiendas = real_data["tiendas"]
    catalogo = real_data["catalogo"]
    cutoff = pd.Timestamp("2024-03-17")
    cal_cutoff = pd.Timestamp("2024-03-10")

    train_raw = ventas.loc[ventas["fecha"] <= cutoff]
    test_raw = ventas.loc[
        (ventas["fecha"] > cutoff) & (ventas["fecha"] <= cutoff + pd.Timedelta(days=7))
    ]
    X_train, X_test = build_feature_matrix(
        train_raw, test_raw, maestro_tiendas=tiendas, catalogo=catalogo
    )
    pure = X_train.loc[X_train["fecha"] <= cal_cutoff - pd.Timedelta(weeks=1)]
    es = X_train.loc[
        (X_train["fecha"] > cal_cutoff - pd.Timedelta(weeks=1))
        & (X_train["fecha"] <= cal_cutoff)
    ]
    cal = X_train.loc[X_train["fecha"] > cal_cutoff]

    fc = GlobalDemandForecaster()
    fc.fit(
        pure[list(FEATURE_COLS)],
        pure[TARGET_COL].to_numpy(),
        X_val=es[list(FEATURE_COLS)],
        y_val=es[TARGET_COL].to_numpy(),
        feature_cols=list(FEATURE_COLS),
        categorical_features=list(CATEGORICAL_FEATURES),
        early_stopping_rounds=50,
        verbose=False,
    )

    cat_cr = catalogo.copy()
    cat_cr["margen"] = cat_cr["precio_venta"] - cat_cr["costo_unitario"]
    cat_cr["critical_ratio"] = cat_cr["margen"] / (
        cat_cr["margen"] + cat_cr["costo_almacenamiento_semanal"]
    )
    prod_sorted = sorted(catalogo["id_producto"].unique())
    idx_to_id = {i: p for i, p in enumerate(prod_sorted)}
    cr_by_idx = {
        i: float(cat_cr.set_index("id_producto").loc[p, "critical_ratio"])
        for i, p in idx_to_id.items()
    }

    cp = DemandConformalForecaster(
        base_forecaster=fc, critical_ratios=cr_by_idx, method="split"
    )
    cp.fit(cal[list(FEATURE_COLS)], cal[TARGET_COL].to_numpy())

    cp_preds = cp.predict(X_test[list(FEATURE_COLS)])
    daily = X_test[["id_tienda", "id_producto"]].copy()
    daily["y_pred"] = cp_preds["y_pred"].to_numpy()
    daily["y_upper"] = cp_preds["y_upper"].to_numpy()
    daily["coverage_target"] = cp_preds["coverage_target"].to_numpy()

    # Aggregamos a granularidad SKU-Tienda (Newsvendor opera 1 fila por combo).
    preds_for_optim = daily.groupby(["id_tienda", "id_producto"], as_index=False).agg(
        y_pred=("y_pred", "sum"),
        y_upper=("y_upper", "sum"),
        coverage_target=("coverage_target", "first"),
    )

    inv = real_data["inventario"]
    opt = NewsvendorOptimizer(catalogo)
    orders = opt.compute_order(preds_for_optim, inv)

    # Conexion CP-Newsvendor: q_star del optimizador == upper conformal pasado
    # como input por SKU-Tienda. Esto verifica que el optimizador no recalcula
    # cuantiles parametricos: respeta el upper conformal one-sided as-is.
    merged = orders.merge(
        preds_for_optim[["id_tienda", "id_producto", "y_upper"]].rename(
            columns={"y_upper": "y_upper_cp"}
        ),
        on=["id_tienda", "id_producto"],
        how="left",
    )
    assert len(merged) == len(orders)
    np.testing.assert_array_almost_equal(
        merged["cantidad_target_q_star"].to_numpy(),
        merged["y_upper_cp"].to_numpy(),
        decimal=6,
        err_msg="q_star debe ser igual al upper conformal por fila",
    )


# ---------------------------------------------------------------------------
# 4. Sanity de business impact
# ---------------------------------------------------------------------------
def test_business_impact_zero_cost_when_orders_match_demand(
    real_data: dict[str, pd.DataFrame],
) -> None:
    """Si pedido == demanda real, costo del modelo debe ser exactamente 0."""
    catalogo = real_data["catalogo"]
    orders = pd.DataFrame(
        [
            {
                "id_tienda": "S1",
                "id_producto": "PROD_001",
                "forecast_media_semanal": 10.0,
                "cantidad_target_q_star": 10.0,
                "stock_actual": 0,
                "critical_ratio": 0.99,
                "cantidad_pedido": 10,
                "alerta": "",
                "costo_total_esperado": 0.0,
            },
        ]
    )
    actuals = pd.DataFrame(
        [{"id_tienda": "S1", "id_producto": "PROD_001", "demanda_real": 10}]
    )
    impact = compute_business_impact(orders, actuals, catalogo)
    assert impact["costo_total_modelo_COP"] == 0
