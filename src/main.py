"""Pipeline end-to-end de Tostao' Demand Forecasting.

Orquesta:
    1. Carga (loader) y validacion de schemas.
    2. Construccion de la matriz de features (preprocessor).
    3. Split temporal final con el layout cronologico canonico:

           pure_train  ->  es_part (1 sem)  ->  cal_part (3 sem)  ->  semana objetivo

       ``es_part`` se separa explicitamente de ``cal_part`` para que el
       early stopping del LightGBM no contamine los residuos usados por
       Conformal Prediction. Si se mezclaran:
           * El modelo se selecciona minimizando error en cal_part.
           * Los residuos en cal_part subestiman los residuos out-of-sample.
           * El cuantil empirico q_hat sale demasiado tight.
           * Sub-cobertura sistematica del intervalo conformal en validacion.
       Bug encontrado en Fase 3: PROD_007 (Pastel de Pollo) cobertura
       0.9814 vs target 0.9852. Tras separar es_part: 0.9857 (gap +0.0005).

    4. Fit del :class:`GlobalDemandForecaster` con early stopping en es_part.
    5. Calibracion del :class:`DemandConformalForecaster` (split one-sided
       + buffer finita-muestra) sobre cal_part, agregada a granularidad
       semanal por SKU-Tienda.
    6. Prediccion de la semana objetivo: y_pred diario y q_hat semanal.
    7. Optimizacion Newsvendor: Q* = upper conformal semanal por
       SKU-Tienda; pedido = max(0, ceil(Q*) - stock_actual).
    8. Outputs:
           - ``outputs/predictions_week_next.csv``  (granularidad diaria)
           - ``outputs/purchase_orders.csv``        (granularidad semanal)
    9. Resumen ejecutivo + impacto de negocio (modelo vs baseline naive).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# sys.path.insert para soportar tanto `python -m src.main` como `python src/main.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402  (path setup must precede project imports)
    CALIBRATION_WEEKS,
    CATEGORICAL_FEATURES,
    FEATURE_COLS,
    FORECAST_HORIZON_DAYS,
    OUTPUT_DIR,
    RANDOM_SEED,
    TARGET_COL,
)
from src.data.loader import load_all_data  # noqa: E402
from src.data.preprocessor import build_feature_matrix  # noqa: E402
from src.models.conformal import split_conformal_intervals  # noqa: E402
from src.models.forecaster import GlobalDemandForecaster  # noqa: E402
from src.optimization.newsvendor import NewsvendorOptimizer, compute_business_impact  # noqa: E402

logger = logging.getLogger("tostao.main")


# ===========================================================================
# Pipeline
# ===========================================================================
def run_pipeline(
    *,
    output_dir: Path = OUTPUT_DIR,
    forecast_horizon_days: int = FORECAST_HORIZON_DAYS,
    calibration_weeks: int = CALIBRATION_WEEKS,
    random_state: int = RANDOM_SEED,
) -> dict:
    """Ejecuta el pipeline completo y retorna un dict de artefactos.

    Returns:
        Dict con claves:
            ``predictions_daily``: DataFrame por (fecha, tienda, producto).
            ``purchase_orders``: DataFrame por (tienda, producto) con pedido.
            ``business_impact``: Dict de metricas de impacto.
            ``output_paths``: Dict con paths absolutos de los CSV escritos.
    """
    t_start = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # 1. Carga
    # --------------------------------------------------------------
    logger.info("Cargando datos...")
    data = load_all_data()
    ventas = data["ventas"]
    tiendas = data["tiendas"]
    catalogo = data["catalogo"]
    inventario = data["inventario"]

    target_week_end = ventas["fecha"].max()
    target_week_start = target_week_end - pd.Timedelta(days=forecast_horizon_days - 1)
    cutoff = target_week_start - pd.Timedelta(days=1)
    logger.info(
        "Semana objetivo: %s -> %s   |   Train: hasta %s",
        target_week_start.date(),
        target_week_end.date(),
        cutoff.date(),
    )

    train_raw = ventas.loc[ventas["fecha"] <= cutoff].copy()
    next_week_raw = ventas.loc[ventas["fecha"] > cutoff].copy()

    # --------------------------------------------------------------
    # 2. Features
    # --------------------------------------------------------------
    logger.info("Construyendo matriz de features...")
    X_train, X_next = build_feature_matrix(
        train_raw,
        next_week_raw,
        maestro_tiendas=tiendas,
        catalogo=catalogo,
        drop_train_stockouts=True,
        drop_lag_nans=True,
    )
    logger.info("X_train: %s   X_next: %s", X_train.shape, X_next.shape)

    # --------------------------------------------------------------
    # 3. Split temporal canonico:  pure_train | es_part | cal_part | next
    # --------------------------------------------------------------
    cal_cutoff = cutoff - pd.Timedelta(weeks=calibration_weeks)
    es_cutoff = cal_cutoff - pd.Timedelta(weeks=1)
    pure_train = X_train.loc[X_train["fecha"] <= es_cutoff]
    es_part = X_train.loc[
        (X_train["fecha"] > es_cutoff) & (X_train["fecha"] <= cal_cutoff)
    ]
    cal_part = X_train.loc[X_train["fecha"] > cal_cutoff]
    if pure_train.empty or es_part.empty or cal_part.empty:
        raise RuntimeError(
            f"Split degenero: pure_train={len(pure_train)} es={len(es_part)} cal={len(cal_part)}"
        )
    logger.info(
        "Layout (filas):  pure_train=%d  |  es_part=%d  |  cal_part=%d",
        len(pure_train),
        len(es_part),
        len(cal_part),
    )

    # --------------------------------------------------------------
    # 4. Fit del forecaster global con early stopping en es_part
    # --------------------------------------------------------------
    logger.info("Entrenando GlobalDemandForecaster...")
    forecaster = GlobalDemandForecaster(random_state=random_state)
    forecaster.fit(
        pure_train[list(FEATURE_COLS)],
        pure_train[TARGET_COL].to_numpy(),
        X_val=es_part[list(FEATURE_COLS)],
        y_val=es_part[TARGET_COL].to_numpy(),
        feature_cols=list(FEATURE_COLS),
        categorical_features=list(CATEGORICAL_FEATURES),
        early_stopping_rounds=50,
        verbose=False,
    )
    logger.info("best_iteration=%s", forecaster.best_iteration)

    # --------------------------------------------------------------
    # 5. Calibracion conformal a granularidad semanal por SKU-Tienda
    # --------------------------------------------------------------
    cal_with_pred = cal_part.copy()
    cal_with_pred["y_pred"] = forecaster.predict(cal_with_pred[list(FEATURE_COLS)])

    # Marca semana ISO por fila para agregar
    cal_with_pred["iso_week"] = cal_with_pred["fecha"].dt.isocalendar().week.astype(int)
    cal_weekly = cal_with_pred.groupby(
        ["id_tienda", "id_producto", "iso_week"], as_index=False
    ).agg(
        y_true=(TARGET_COL, "sum"),
        y_pred=("y_pred", "sum"),
        id_producto_encoded=("id_producto_encoded", "first"),
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

    q_hat_weekly: dict[int, float] = {}
    for idx_p, cr in cr_by_idx.items():
        sub = cal_weekly[cal_weekly["id_producto_encoded"] == idx_p]
        if sub.empty:
            logger.warning("Sin calibracion para producto idx=%d", idx_p)
            continue
        _, _, q_hat = split_conformal_intervals(
            sub["y_true"].to_numpy(),
            sub["y_pred"].to_numpy(),
            np.zeros(1),
            alpha=1.0 - cr,
            one_sided=True,
            finite_sample_buffer=True,
        )
        q_hat_weekly[idx_p] = q_hat
        logger.info(
            "Producto %s (CR=%.4f, n_cal=%d weeks*stores)  ->  q_hat_weekly=%.2f",
            idx_to_id[idx_p],
            cr,
            len(sub),
            q_hat,
        )

    # --------------------------------------------------------------
    # 6. Prediccion semana objetivo (diaria) + agregacion semanal
    # --------------------------------------------------------------
    logger.info("Prediciendo semana objetivo...")
    X_next = X_next.copy()
    X_next["y_pred"] = forecaster.predict(X_next[list(FEATURE_COLS)])

    # build_feature_matrix no propaga el target al test (para inferencia real
    # no tendriamos targets futuros). Para reportar impacto contra la realidad
    # observada, mergeamos los actuals de next_week_raw para evaluacion.
    daily_predictions = (
        X_next[
            [
                "fecha",
                "id_tienda",
                "id_producto",
                "id_producto_encoded",
                "y_pred",
            ]
        ]
        .merge(
            next_week_raw[["fecha", "id_tienda", "id_producto", TARGET_COL]],
            on=["fecha", "id_tienda", "id_producto"],
            how="left",
        )
        .rename(columns={TARGET_COL: "y_true"})
    )
    daily_predictions["coverage_target"] = daily_predictions["id_producto_encoded"].map(
        cr_by_idx
    )

    weekly_pred = daily_predictions.groupby(
        ["id_tienda", "id_producto"], as_index=False
    ).agg(
        forecast_media_semanal=("y_pred", "sum"),
        demanda_real=("y_true", "sum"),
        id_producto_encoded=("id_producto_encoded", "first"),
    )
    weekly_pred["q_hat_weekly"] = weekly_pred["id_producto_encoded"].map(q_hat_weekly)
    weekly_pred["y_pred"] = weekly_pred["forecast_media_semanal"]
    weekly_pred["y_upper"] = weekly_pred["y_pred"] + weekly_pred["q_hat_weekly"]
    weekly_pred["y_lower"] = 0.0
    weekly_pred["coverage_target"] = weekly_pred["id_producto_encoded"].map(cr_by_idx)

    # --------------------------------------------------------------
    # 7. Newsvendor
    # --------------------------------------------------------------
    logger.info("Aplicando NewsvendorOptimizer...")
    optimizer = NewsvendorOptimizer(catalogo)
    purchase_orders = optimizer.compute_order(
        weekly_pred[
            ["id_tienda", "id_producto", "y_pred", "y_upper", "coverage_target"]
        ],
        inventario,
    )

    # --------------------------------------------------------------
    # 8. Outputs CSV
    # --------------------------------------------------------------
    pred_path = output_dir / "predictions_week_next.csv"
    orders_path = output_dir / "purchase_orders.csv"

    daily_out = daily_predictions.copy()
    daily_out["q_hat_weekly_producto"] = daily_out["id_producto_encoded"].map(
        q_hat_weekly
    )
    daily_out = daily_out.rename(columns={"y_true": "demanda_real"})
    daily_out = daily_out[
        [
            "fecha",
            "id_tienda",
            "id_producto",
            "y_pred",
            "demanda_real",
            "coverage_target",
            "q_hat_weekly_producto",
        ]
    ]
    daily_out.to_csv(pred_path, index=False)
    purchase_orders.to_csv(orders_path, index=False)
    logger.info("Escrito: %s (%d filas)", pred_path, len(daily_out))
    logger.info("Escrito: %s (%d filas)", orders_path, len(purchase_orders))

    # --------------------------------------------------------------
    # 9. Impacto de negocio
    # --------------------------------------------------------------
    actual_weekly = next_week_raw.groupby(
        ["id_tienda", "id_producto"], as_index=False
    ).agg(demanda_real=(TARGET_COL, "sum"))
    impact = compute_business_impact(purchase_orders, actual_weekly, catalogo)

    elapsed = time.perf_counter() - t_start
    impact["pipeline_seconds"] = round(elapsed, 1)

    return {
        "predictions_daily": daily_out,
        "purchase_orders": purchase_orders,
        "business_impact": impact,
        "output_paths": {"predictions": pred_path, "orders": orders_path},
    }


def print_summary(impact: dict, purchase_orders: pd.DataFrame) -> None:
    """Resumen ejecutivo en consola."""
    detail = impact.pop("_detail_df", None)
    separator = "=" * 72
    print()
    print(separator)
    print("RESUMEN EJECUTIVO  -  Tostao' Demand Forecasting & Supply Optimization")
    print(separator)

    print("\n[Pedido recomendado]")
    print(f"  Combos SKU-Tienda evaluados : {impact['n_sku_tienda_evaluados']}")
    print(f"  Total unidades a pedir      : {impact['pedido_modelo_total']}")
    print(f"  Total unidades pedido naive : {impact['pedido_naive_total']}")
    n_sobrestock = int((purchase_orders["alerta"] != "").sum())
    print(f"  Combos con alerta SOBRESTOCK: {n_sobrestock}")

    print("\n[Pedido por producto]")
    by_prod = (
        purchase_orders.groupby("id_producto")
        .agg(
            n_combos=("id_tienda", "count"),
            total_pedido=("cantidad_pedido", "sum"),
            avg_q_star=("cantidad_target_q_star", "mean"),
        )
        .round(1)
    )
    print(by_prod.to_string())

    print("\n[Costos realizados (semana objetivo)]")
    print(f"  Costo total modelo  : {impact['costo_total_modelo_COP']:>14,} COP")
    print(f"  Costo total naive   : {impact['costo_total_naive_COP']:>14,} COP")
    print(f"  Ahorro absoluto     : {impact['ahorro_COP']:>14,} COP")
    print(f"  Ahorro porcentual   : {impact['ahorro_pct']:>14.2f} %")
    print()
    print("  Decomposicion del ahorro:")
    print(
        f"    Via menos stockout : {impact['ahorro_via_menos_stockout_COP']:>14,} COP"
    )
    print(
        f"    Via menos overstock: {impact['ahorro_via_menos_overstock_COP']:>14,} COP"
    )
    print()
    print("  Tasas de error (combos / total):")
    print(
        f"    Stockout modelo  : {impact['pct_combos_con_stockout_modelo']:.1%}"
        f"   (naive: {impact['pct_combos_con_stockout_naive']:.1%})"
    )
    print(
        f"    Overstock modelo : {impact['pct_combos_con_overstock_modelo']:.1%}"
        f"   (naive: {impact['pct_combos_con_overstock_naive']:.1%})"
    )
    print()
    print(f"Pipeline ejecutado en {impact['pipeline_seconds']}s.")
    print(separator)
    if detail is not None:
        impact["_detail_df"] = detail


# ===========================================================================
# CLI
# ===========================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline end-to-end Tostao' demand forecasting & supply optimization.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Carpeta de salida (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=FORECAST_HORIZON_DAYS,
        help=f"Horizonte de pronostico en dias (default: {FORECAST_HORIZON_DAYS})",
    )
    parser.add_argument(
        "--calibration-weeks",
        type=int,
        default=CALIBRATION_WEEKS,
        help=f"Semanas para calibracion conformal (default: {CALIBRATION_WEEKS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    result = run_pipeline(
        output_dir=args.output_dir,
        forecast_horizon_days=args.horizon,
        calibration_weeks=args.calibration_weeks,
        random_state=args.seed,
    )
    print_summary(result["business_impact"], result["purchase_orders"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
