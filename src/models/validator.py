"""Validacion cruzada temporal estricta para el forecaster + conformal.

Proporciona :func:`cross_validate_forecaster`, que itera sobre ``n_splits``
folds expanding-window por **semana** (no por filas). Para cada fold:

    1. Construye la matriz de features con :func:`build_feature_matrix`
       (los lags se calculan DENTRO del fold; las primeras 14 obs por combo
       se descartan, los stockouts implicitos se filtran del target).
    2. Reserva las ultimas ``calibration_weeks`` del train para calibrar el
       conformal y para early stopping.
    3. Entrena :class:`GlobalDemandForecaster` con early stopping en el
       calibration set.
    4. Ajusta :class:`DemandConformalForecaster` (Split por defecto) sobre
       el calibration set, con ``confidence_level = critical_ratio``
       por producto.
    5. Predice en validacion y reporta MAE, RMSE, WAPE, cobertura empirica,
       ancho promedio del intervalo, Winkler score, y best_iteration.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import (
    CALIBRATION_WEEKS,
    CATEGORICAL_FEATURES,
    FEATURE_COLS,
    MIN_TRAIN_WEEKS,
    RANDOM_SEED,
    TARGET_COL,
)
from src.data.preprocessor import build_feature_matrix
from src.models.conformal import (
    DemandConformalForecaster,
    winkler_score,
)
from src.models.forecaster import GlobalDemandForecaster

logger = logging.getLogger(__name__)


# ===========================================================================
# Splits por semana
# ===========================================================================
def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def panel_time_series_splits(
    ventas: pd.DataFrame,
    *,
    n_splits: int = 5,
    forecast_horizon_days: int = 7,
    min_train_weeks: int = MIN_TRAIN_WEEKS,
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Genera ``n_splits`` ventanas expanding-window de horizonte semanal.

    Yields:
        Tripletas ``(train_start, train_end, val_end)`` (inclusive) tales que
        el train cubre ``[train_start, train_end]`` y la validacion cubre
        ``(train_end, val_end]``.
    """
    fechas = pd.Series(sorted(ventas["fecha"].unique()))
    total_days = (fechas.iloc[-1] - fechas.iloc[0]).days + 1
    if total_days < min_train_weeks * 7 + n_splits * forecast_horizon_days:
        raise ValueError(
            f"Datos insuficientes: {total_days} dias para {min_train_weeks} "
            f"semanas de train + {n_splits} folds x {forecast_horizon_days} dias."
        )
    train_start = fechas.iloc[0]
    last_val_end = fechas.iloc[-1]
    for fold in range(n_splits):
        offset_back = (n_splits - fold - 1) * forecast_horizon_days
        val_end = last_val_end - pd.Timedelta(days=offset_back)
        train_end = val_end - pd.Timedelta(days=forecast_horizon_days)
        yield train_start, train_end, val_end


# ===========================================================================
# Estructura para resultado por fold
# ===========================================================================
@dataclass
class FoldResult:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    n_train: int
    n_cal: int
    n_val: int
    mae: float
    rmse: float
    wape: float
    coverage_achieved: float
    coverage_target_avg: float
    avg_interval_width: float
    winkler: float
    best_iteration: int | None
    fit_seconds: float
    fallback_used: bool

    def to_dict(self) -> dict:
        return {
            "fold": self.fold,
            "train_window": f"{self.train_start.date()} -> {self.train_end.date()}",
            "val_end": self.val_end.date().isoformat(),
            "n_train": self.n_train,
            "n_cal": self.n_cal,
            "n_val": self.n_val,
            "MAE": round(self.mae, 3),
            "RMSE": round(self.rmse, 3),
            "WAPE": round(self.wape, 4),
            "coverage_achieved": round(self.coverage_achieved, 4),
            "coverage_target_avg": round(self.coverage_target_avg, 4),
            "avg_interval_width": round(self.avg_interval_width, 3),
            "winkler": round(self.winkler, 3),
            "best_iter": self.best_iteration,
            "fit_seconds": round(self.fit_seconds, 1),
            "fallback": self.fallback_used,
        }


# ===========================================================================
# Cross-validation
# ===========================================================================
def cross_validate_forecaster(
    ventas: pd.DataFrame,
    maestro_tiendas: pd.DataFrame,
    catalogo: pd.DataFrame,
    *,
    n_splits: int = 5,
    forecast_horizon_days: int = 7,
    calibration_weeks: int = CALIBRATION_WEEKS,
    method: str = DemandConformalForecaster.METHOD_SPLIT,
    feature_cols: list[str] | None = None,
    categorical_features: list[str] | None = None,
    return_predictions: bool = False,
    random_state: int = RANDOM_SEED,
) -> dict:
    """Cross-validation temporal completa con CP por producto.

    Args:
        ventas: DataFrame de ventas historicas (sin features).
        maestro_tiendas: Catalogo de tiendas (universo para encoding).
        catalogo: Catalogo de productos (incluye CR derivado).
        n_splits: Numero de folds.
        forecast_horizon_days: Horizonte de cada validacion.
        calibration_weeks: Semanas finales del train usadas para calibrar CP.
        method: ``'split'`` (default) o ``'enbpi'``.
        feature_cols: Override de features. Default :data:`FEATURE_COLS`.
        categorical_features: Override de cat features.
            Default :data:`CATEGORICAL_FEATURES`.
        return_predictions: Si True, agrega un DataFrame ``predictions`` con
            todas las predicciones por fold (util para plotting).

    Returns:
        Dict con:
            ``metrics``: DataFrame de metricas por fold + columna 'avg'.
            ``coverage_per_product``: DataFrame consolidado (filas = producto,
                cols = coverage_achieved/target/gap/avg_width agregado entre folds).
            ``feature_importance``: Series promedio entre folds.
            ``predictions``: opcional, DataFrame con predicciones detalladas.
    """
    feature_cols = list(feature_cols) if feature_cols else list(FEATURE_COLS)
    categorical_features = (
        list(categorical_features)
        if categorical_features
        else list(CATEGORICAL_FEATURES)
    )

    # Critical ratio por producto (encoded)
    cat_df = catalogo.copy()
    cat_df["margen"] = cat_df["precio_venta"] - cat_df["costo_unitario"]
    cat_df["critical_ratio"] = cat_df["margen"] / (
        cat_df["margen"] + cat_df["costo_almacenamiento_semanal"]
    )
    prod_sorted = sorted(cat_df["id_producto"].unique())
    prod_to_idx = {p: i for i, p in enumerate(prod_sorted)}
    cr_by_idx = {
        prod_to_idx[p]: float(cr)
        for p, cr in zip(cat_df["id_producto"], cat_df["critical_ratio"])
    }

    fold_records: list[FoldResult] = []
    coverage_records: list[pd.DataFrame] = []
    importance_records: list[pd.Series] = []
    prediction_records: list[pd.DataFrame] = []
    fallback_any = False

    splits = list(
        panel_time_series_splits(
            ventas,
            n_splits=n_splits,
            forecast_horizon_days=forecast_horizon_days,
        )
    )

    for fold_idx, (train_start, train_end, val_end) in enumerate(splits, start=1):
        t0 = time.perf_counter()
        train_raw = ventas.loc[
            (ventas["fecha"] >= train_start) & (ventas["fecha"] <= train_end)
        ].copy()
        val_raw = ventas.loc[
            (ventas["fecha"] > train_end) & (ventas["fecha"] <= val_end)
        ].copy()

        X_train, X_val = build_feature_matrix(
            train_raw,
            val_raw,
            maestro_tiendas=maestro_tiendas,
            catalogo=catalogo,
            drop_train_stockouts=True,
            drop_lag_nans=True,
        )

        # Estructura del fold (cronologica):
        #   pure_train  ->  es_part (1 sem)  ->  cal_part (3 sem)  ->  val (1 sem)
        # Usamos es_part para early stopping y cal_part SOLO para conformal.
        # Asi los residuos de calibracion son out-of-sample respecto al modelo.
        cal_cutoff = train_end - pd.Timedelta(weeks=calibration_weeks)
        es_cutoff = cal_cutoff - pd.Timedelta(weeks=1)
        cal_part = X_train.loc[X_train["fecha"] > cal_cutoff].copy()
        es_part = X_train.loc[
            (X_train["fecha"] > es_cutoff) & (X_train["fecha"] <= cal_cutoff)
        ].copy()
        pure_train = X_train.loc[X_train["fecha"] <= es_cutoff].copy()
        if cal_part.empty or pure_train.empty or es_part.empty:
            raise RuntimeError(
                f"Fold {fold_idx}: split degenero "
                f"(pure_train {len(pure_train)} / es {len(es_part)} / cal {len(cal_part)})."
            )

        y_train = pure_train[TARGET_COL].to_numpy()
        y_es = es_part[TARGET_COL].to_numpy()
        y_cal = cal_part[TARGET_COL].to_numpy()
        y_val = val_raw[TARGET_COL].to_numpy()

        forecaster = GlobalDemandForecaster(
            objective="regression",
            random_state=random_state,
        )
        forecaster.fit(
            pure_train[feature_cols],
            y_train,
            X_val=es_part[feature_cols],
            y_val=y_es,
            feature_cols=feature_cols,
            categorical_features=categorical_features,
            early_stopping_rounds=50,
            verbose=False,
        )

        cp = DemandConformalForecaster(
            base_forecaster=forecaster,
            critical_ratios=cr_by_idx,
            method=method,
            calibration_weeks=calibration_weeks,
            random_state=random_state,
        )
        cp.fit(cal_part[feature_cols], y_cal)
        if cp.fallback_used:
            fallback_any = True

        # Predicciones puntuales para metricas tecnicas
        y_pred_val = forecaster.predict(X_val[feature_cols])

        # Intervalos conformales
        cp_preds = cp.predict(X_val[feature_cols])
        cp_preds = cp_preds.reindex(X_val.index)

        coverage_summary = cp.evaluate_coverage(X_val[feature_cols], y_val)
        coverage_summary["fold"] = fold_idx
        coverage_records.append(coverage_summary.reset_index())

        mae = float(mean_absolute_error(y_val, y_pred_val))
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
        wape = _wape(y_val, y_pred_val)

        # Cobertura agregada (todos los productos juntos)
        in_interval = (y_val >= cp_preds["y_lower"].to_numpy()) & (
            y_val <= cp_preds["y_upper"].to_numpy()
        )
        coverage_global = float(in_interval.mean())
        avg_width = float((cp_preds["y_upper"] - cp_preds["y_lower"]).mean())
        # Winkler ponderado por producto-CR (cada fila usa su alpha)
        wk_per_row = []
        for prod, sub in cp_preds.groupby("id_producto_encoded"):
            cr_p = cr_by_idx[int(prod)]
            mask = sub.index
            wk = winkler_score(
                y_val[X_val.index.get_indexer(mask)],
                sub["y_lower"].to_numpy(),
                sub["y_upper"].to_numpy(),
                alpha=1.0 - cr_p,
            )
            wk_per_row.append((wk, len(sub)))
        winkler_global = float(
            np.average([w for w, _ in wk_per_row], weights=[n for _, n in wk_per_row])
        )
        coverage_target_avg = float(
            np.average(
                cp_preds["coverage_target"].to_numpy(),
                weights=np.ones(len(cp_preds)),
            )
        )

        importance_records.append(
            forecaster.get_feature_importance().rename(f"fold_{fold_idx}")
        )

        if return_predictions:
            preds_df = X_val[["fecha", "id_tienda", "id_producto"]].copy()
            preds_df["y_true"] = y_val
            preds_df["y_pred"] = y_pred_val
            preds_df["y_lower"] = cp_preds["y_lower"].to_numpy()
            preds_df["y_upper"] = cp_preds["y_upper"].to_numpy()
            preds_df["coverage_target"] = cp_preds["coverage_target"].to_numpy()
            preds_df["fold"] = fold_idx
            prediction_records.append(preds_df)

        fold_results = FoldResult(
            fold=fold_idx,
            train_start=train_start,
            train_end=train_end,
            val_end=val_end,
            n_train=len(pure_train),
            n_cal=len(cal_part),
            n_val=len(X_val),
            mae=mae,
            rmse=rmse,
            wape=wape,
            coverage_achieved=coverage_global,
            coverage_target_avg=coverage_target_avg,
            avg_interval_width=avg_width,
            winkler=winkler_global,
            best_iteration=forecaster.best_iteration,
            fit_seconds=time.perf_counter() - t0,
            fallback_used=cp.fallback_used,
        )
        fold_records.append(fold_results)
        logger.info("Fold %d: %s", fold_idx, fold_results.to_dict())

    metrics_df = pd.DataFrame([fr.to_dict() for fr in fold_records])
    numeric_cols = [
        "MAE",
        "RMSE",
        "WAPE",
        "coverage_achieved",
        "coverage_target_avg",
        "avg_interval_width",
        "winkler",
    ]
    avg_row = metrics_df[numeric_cols].mean().to_dict()
    avg_row.update(
        {
            "fold": "avg",
            "train_window": "-",
            "val_end": "-",
            "n_train": metrics_df["n_train"].sum(),
            "n_cal": metrics_df["n_cal"].sum(),
            "n_val": metrics_df["n_val"].sum(),
            "best_iter": metrics_df["best_iter"].mean(),
            "fit_seconds": metrics_df["fit_seconds"].sum(),
            "fallback": metrics_df["fallback"].any(),
        }
    )
    metrics_df = pd.concat([metrics_df, pd.DataFrame([avg_row])], ignore_index=True)
    for c in numeric_cols + ["best_iter", "fit_seconds"]:
        metrics_df[c] = pd.to_numeric(metrics_df[c], errors="coerce").round(4)

    coverage_df = pd.concat(coverage_records, ignore_index=True)
    importance_df = pd.concat(importance_records, axis=1)
    importance_avg = importance_df.mean(axis=1).sort_values(ascending=False)

    # Cobertura consolidada por producto a traves de folds (ponderada por n)
    agg_cov = (
        coverage_df.assign(weight=lambda d: d["n"])
        .groupby("id_producto_encoded")
        .apply(
            lambda d: pd.Series(
                {
                    "n_total": int(d["n"].sum()),
                    "coverage_target": float(d["coverage_target"].iloc[0]),
                    "coverage_achieved": float(
                        np.average(d["coverage_achieved"], weights=d["n"])
                    ),
                    "avg_width": float(np.average(d["avg_width"], weights=d["n"])),
                }
            )
        )
    )
    agg_cov["gap"] = agg_cov["coverage_achieved"] - agg_cov["coverage_target"]

    out: dict = {
        "metrics": metrics_df,
        "coverage_per_product": agg_cov.sort_index(),
        "coverage_per_fold": coverage_df,
        "feature_importance": importance_avg,
        "fallback_used": fallback_any,
    }
    if return_predictions:
        out["predictions"] = pd.concat(prediction_records, ignore_index=True)
    return out
