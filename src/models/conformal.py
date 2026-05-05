"""Conformal Prediction para demanda con cobertura por producto = critical ratio.

"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import CALIBRATION_WEEKS, RANDOM_SEED
from src.models.forecaster import GlobalDemandForecaster

logger = logging.getLogger(__name__)


# ===========================================================================
# Implementacion canonica: Split Conformal one-sided con buffer finito-muestra
# ===========================================================================
def split_conformal_intervals(
    y_cal_true: np.ndarray,
    y_cal_pred: np.ndarray,
    y_test_pred: np.ndarray,
    alpha: float,
    *,
    one_sided: bool = True,
    finite_sample_buffer: bool = True,
    nonneg_lower: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Intervalos conformales por Split Conformal.

    Implementacion canonica del proyecto (one_sided=True): score de
    no-conformidad = residuo firmado ``s_i = y_i - hat_y_i`` (sin valor
    absoluto). El upper bound se construye como ``hat_y + q_hat`` donde
    ``q_hat`` es el cuantil ``1 - effective_alpha`` de los scores. El
    lower bound se fija en 0.

    Justificacion (decision documentada en el skill canonico):
        Newsvendor solo necesita controlar P(D > upper_bound) <= 1 - CR.
        Un score asimetrico (residuo signed) es teoricamente optimo para
        este problema vs el score absoluto simetrico, que sobreestima la
        incertidumbre del lado izquierdo y reparte la cobertura
        desperdiciando masa en la cola inferior.

    Buffer finito-muestra (``finite_sample_buffer=True``):
        ``effective_alpha = max(alpha - 1/n_cal, 1e-4)``. Absorbe la
        varianza del cuantil empirico cuando n_cal es modesto.

    Para series de tiempo el calibration set DEBE ser cronologicamente
    posterior al training y previo al test; el caller es responsable.

    Args:
        y_cal_true: Demanda real en calibracion.
        y_cal_pred: Prediccion del modelo en calibracion.
        y_test_pred: Prediccion del modelo en test.
        alpha: ``1 - coverage_level`` (ej. ``alpha=0.006`` para CR=0.994).
        one_sided: Si True (default), usa la implementacion canonica del
            proyecto. Si False, comportamiento legacy con ``|y - hat_y|``
            simetrico (deja el lower bound real en lugar de 0).
        finite_sample_buffer: Si True, aplica el buffer ``1/n_cal``.
        nonneg_lower: En modo simetrico, recortar el lower a 0. Ignorado
            en modo one-sided porque el lower es siempre 0 por diseno.

    Returns:
        Tupla ``(lower, upper, q_hat)`` con ``q_hat`` = cuantil del score.
    """
    if len(y_cal_true) != len(y_cal_pred):
        raise ValueError("y_cal_true y y_cal_pred deben tener la misma longitud.")
    n = len(y_cal_true)
    if n == 0:
        raise ValueError("Calibration set vacio.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha debe estar en (0,1), recibi {alpha}")

    eff_alpha = max(alpha - 1.0 / n, 1e-4) if finite_sample_buffer else alpha
    q_level = float(np.clip(np.ceil((n + 1) * (1.0 - eff_alpha)) / n, 0.0, 1.0))

    y_cal_true_arr = np.asarray(y_cal_true, dtype=float)
    y_cal_pred_arr = np.asarray(y_cal_pred, dtype=float)
    y_test_pred_arr = np.asarray(y_test_pred, dtype=float)

    if one_sided:
        scores = y_cal_true_arr - y_cal_pred_arr  # firmado, asimetrico
        q_hat = float(np.quantile(scores, q_level, method="higher"))
        upper = y_test_pred_arr + q_hat
        lower = np.zeros_like(y_test_pred_arr)
    else:
        scores = np.abs(y_cal_true_arr - y_cal_pred_arr)
        q_hat = float(np.quantile(scores, q_level, method="higher"))
        upper = y_test_pred_arr + q_hat
        lower = y_test_pred_arr - q_hat
        if nonneg_lower:
            lower = np.maximum(0.0, lower)

    return lower, upper, q_hat


# ===========================================================================
# Forecaster conformal por producto
# ===========================================================================
@dataclass
class _ProductCalibration:
    """Estado de calibracion para un producto."""

    product_id: int
    critical_ratio: float
    method: str  # 'split' | 'enbpi'
    q_hat: float | None = None  # solo para split
    mapie: object | None = None  # solo para enbpi
    feature_cols_drop: list[str] = field(default_factory=list)


class DemandConformalForecaster:
    """Wrapper de Conformal Prediction sobre un :class:`GlobalDemandForecaster`.

    Args:
        base_forecaster: Forecaster global ya AJUSTADO. El wrapper solo
            calibra intervalos sobre sus predicciones.
        critical_ratios: ``{id_producto_encoded: cr}`` por producto.
        method: ``'split'`` (default, recomendado para arquitectura global) o
            ``'enbpi'`` (entrena MAPIE :class:`TimeSeriesRegressor` por
            producto; rompe el modelo global).
        cv_blocks: Numero de re-muestreos para BlockBootstrap (solo enbpi).
        random_state: Semilla.

    Notes:
        ``predict`` opera por producto: enmascara X por
        ``id_producto_encoded`` y aplica la calibracion de ese producto.
    """

    METHOD_SPLIT = "split"
    METHOD_ENBPI = "enbpi"

    def __init__(
        self,
        base_forecaster: GlobalDemandForecaster,
        critical_ratios: dict[int | str, float],
        method: str = METHOD_SPLIT,
        cv_blocks: int = 10,
        calibration_weeks: int = CALIBRATION_WEEKS,
        random_state: int = RANDOM_SEED,
    ) -> None:
        if method not in {self.METHOD_SPLIT, self.METHOD_ENBPI}:
            raise ValueError(f"method invalido: {method}")
        if not base_forecaster.is_fitted and method == self.METHOD_SPLIT:
            raise RuntimeError(
                "Para method='split', el base_forecaster debe estar AJUSTADO."
            )

        self.base = base_forecaster
        self.critical_ratios = {int(k): float(v) for k, v in critical_ratios.items()}
        self.method = method
        self.cv_blocks = cv_blocks
        self.calibration_weeks = calibration_weeks
        self.random_state = random_state

        self._calibrations: dict[int, _ProductCalibration] = {}
        self._is_calibrated = False
        self._fallback_used = False

    # ------------------------------------------------------------------
    def fit(
        self,
        X_cal: pd.DataFrame,
        y_cal: pd.Series | np.ndarray,
        id_producto_col: str = "id_producto_encoded",
    ) -> "DemandConformalForecaster":
        """Calibra el wrapper conformal sobre el set de calibracion.

        Para ``method='split'``: por producto, computa ``q_hat`` (cuantil de
        residuos absolutos) en X_cal usando ``base_forecaster.predict``.

        Para ``method='enbpi'``: por producto, entrena un nuevo
        :class:`mapie.regression.TimeSeriesRegressor` con BlockBootstrap sobre
        el subset del producto en X_cal (no usa el base global).

        Args:
            X_cal: Features de calibracion.
            y_cal: Target real de calibracion.
            id_producto_col: Columna con el ID encoded del producto.

        Returns:
            self.
        """
        if id_producto_col not in X_cal.columns:
            raise KeyError(f"X_cal debe tener la columna {id_producto_col!r}.")

        y_cal_arr = np.asarray(y_cal, dtype=float)
        products = sorted(int(p) for p in X_cal[id_producto_col].unique())

        if self.method == self.METHOD_ENBPI:
            try:
                self._fit_enbpi(X_cal, y_cal_arr, products, id_producto_col)
            except Exception as e:  # noqa: BLE001
                warnings.warn(
                    f"MAPIE EnbPI fallo ({type(e).__name__}: {e!s}). "
                    f"Fallback automatico a Split Conformal.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                logger.warning("MAPIE fallback triggered: %s", e)
                self._fallback_used = True
                self._fit_split(X_cal, y_cal_arr, products, id_producto_col)
        else:
            self._fit_split(X_cal, y_cal_arr, products, id_producto_col)

        self._is_calibrated = True
        return self

    # ------------------------------------------------------------------
    def _fit_split(
        self,
        X_cal: pd.DataFrame,
        y_cal: np.ndarray,
        products: list[int],
        id_col: str,
    ) -> None:
        for prod in products:
            cr = self.critical_ratios.get(prod)
            if cr is None:
                raise KeyError(f"Critical ratio no definido para producto {prod}")
            mask = X_cal[id_col].to_numpy() == prod
            if not mask.any():
                continue
            y_pred = self.base.predict(X_cal.loc[mask])
            y_true = y_cal[mask]

            alpha = 1.0 - cr
            _, _, q_hat = split_conformal_intervals(
                y_true,
                y_pred,
                np.zeros(1),
                alpha=alpha,
                one_sided=True,
                finite_sample_buffer=True,
            )
            self._calibrations[prod] = _ProductCalibration(
                product_id=prod,
                critical_ratio=cr,
                method=self.METHOD_SPLIT,
                q_hat=q_hat,
            )

    def _fit_enbpi(
        self,
        X_cal: pd.DataFrame,
        y_cal: np.ndarray,
        products: list[int],
        id_col: str,
    ) -> None:
        from mapie.regression import TimeSeriesRegressor
        from mapie.subsample import BlockBootstrap
        from lightgbm import LGBMRegressor

        for prod in products:
            cr = self.critical_ratios.get(prod)
            if cr is None:
                raise KeyError(f"Critical ratio no definido para producto {prod}")
            mask = X_cal[id_col].to_numpy() == prod
            if not mask.any():
                continue
            X_p = X_cal.loc[mask].drop(columns=[id_col])
            y_p = y_cal[mask]
            if len(y_p) < self.cv_blocks * 2:
                raise ValueError(
                    f"Producto {prod}: {len(y_p)} obs es insuficiente para "
                    f"BlockBootstrap con {self.cv_blocks} bloques."
                )
            cv = BlockBootstrap(
                n_resamplings=self.cv_blocks,
                n_blocks=self.calibration_weeks,
                overlapping=True,
                random_state=self.random_state,
            )
            base = LGBMRegressor(
                **GlobalDemandForecaster.LGBM_BASE_PARAMS,
                random_state=self.random_state,
                objective="regression",
            )
            mapie = TimeSeriesRegressor(
                estimator=base,
                method="enbpi",
                cv=cv,
                agg_function="mean",
                n_jobs=-1,
                random_state=self.random_state,
            )
            mapie.fit(X_p.to_numpy(), y_p)
            self._calibrations[prod] = _ProductCalibration(
                product_id=prod,
                critical_ratio=cr,
                method=self.METHOD_ENBPI,
                mapie=mapie,
                feature_cols_drop=[id_col],
            )

    # ------------------------------------------------------------------
    def predict(
        self,
        X_test: pd.DataFrame,
        id_producto_col: str = "id_producto_encoded",
    ) -> pd.DataFrame:
        """Predicciones puntuales + intervalos conformales por producto.

        Returns:
            DataFrame indexado por el indice de ``X_test`` con columnas
            ``y_pred``, ``y_lower``, ``y_upper``, ``coverage_target``,
            ``id_producto_encoded``.
        """
        self._assert_calibrated()
        rows: list[pd.DataFrame] = []
        for prod, cal in self._calibrations.items():
            mask = X_test[id_producto_col].to_numpy() == prod
            if not mask.any():
                continue
            sub = X_test.loc[mask]
            if cal.method == self.METHOD_SPLIT:
                y_pred = self.base.predict(sub)
                # One-sided: el upper es y_pred + q_hat, el lower es 0 (no
                # se necesita para Newsvendor). Aunque q_hat puede ser
                # ligeramente negativo si el modelo sobre-predice
                # consistentemente, el upper sigue siendo monotonico vs
                # y_pred.
                upper = y_pred + cal.q_hat
                lower = np.zeros_like(y_pred)
            else:  # enbpi
                X_p = sub.drop(columns=cal.feature_cols_drop)
                y_pred, y_pis = cal.mapie.predict(  # type: ignore[union-attr]
                    X_p.to_numpy(), confidence_level=cal.critical_ratio
                )
                y_pred = np.maximum(0.0, np.asarray(y_pred, dtype=float))
                lower = np.maximum(0.0, np.asarray(y_pis[:, 0, 0], dtype=float))
                upper = np.asarray(y_pis[:, 1, 0], dtype=float)
            rows.append(
                pd.DataFrame(
                    {
                        "y_pred": y_pred,
                        "y_lower": lower,
                        "y_upper": upper,
                        "coverage_target": cal.critical_ratio,
                        "id_producto_encoded": prod,
                    },
                    index=sub.index,
                )
            )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "y_pred",
                    "y_lower",
                    "y_upper",
                    "coverage_target",
                    "id_producto_encoded",
                ]
            )
        return pd.concat(rows).sort_index()

    # ------------------------------------------------------------------
    def evaluate_coverage(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        id_producto_col: str = "id_producto_encoded",
    ) -> pd.DataFrame:
        """Cobertura empirica + ancho promedio + gap por producto.

        Returns:
            DataFrame indexado por ``id_producto_encoded`` con columnas:
            ``n``, ``coverage_target``, ``coverage_achieved``, ``gap``,
            ``avg_width``.
        """
        preds = self.predict(X, id_producto_col)
        y_arr = np.asarray(y, dtype=float)
        preds = preds.assign(
            y_real=y_arr[X.index.get_indexer(preds.index)],
        )
        preds["in_interval"] = (preds["y_real"] >= preds["y_lower"]) & (
            preds["y_real"] <= preds["y_upper"]
        )
        preds["width"] = preds["y_upper"] - preds["y_lower"]

        summary = preds.groupby("id_producto_encoded").agg(
            n=("in_interval", "size"),
            coverage_target=("coverage_target", "first"),
            coverage_achieved=("in_interval", "mean"),
            avg_width=("width", "mean"),
        )
        summary["gap"] = summary["coverage_achieved"] - summary["coverage_target"]
        violations = summary[summary["gap"] < 0]
        if not violations.empty:
            logger.warning(
                "Cobertura por debajo del target en %d producto(s): %s",
                len(violations),
                list(violations.index),
            )
        return summary

    # ------------------------------------------------------------------
    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @property
    def fallback_used(self) -> bool:
        return self._fallback_used

    def _assert_calibrated(self) -> None:
        if not self._is_calibrated:
            raise RuntimeError("Conformal no calibrado. Llamar .fit() primero.")


# ===========================================================================
# Metricas estandar de evaluacion del CP
# ===========================================================================
def winkler_score(
    y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> float:
    """Winkler Score (interval score). Penaliza ancho + violaciones.

    Definicion estandar: para cada punto, score = (upper - lower) +
    (2/alpha) * (lower - y) si y < lower, o (2/alpha) * (y - upper) si y >
    upper, sino score = upper - lower. Mas bajo = mejor.
    """
    y = np.asarray(y, dtype=float)
    width = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
    penalty = np.where(
        y < lower,
        (lower - y) * 2.0 / alpha,
        np.where(y > upper, (y - upper) * 2.0 / alpha, 0.0),
    )
    return float(np.mean(width + penalty))


def evaluate_cp_metrics(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    coverage_target: float,
) -> dict[str, float]:
    """Calcula coverage, gap, ancho promedio y Winkler score."""
    y_true = np.asarray(y_true, dtype=float)
    y_lower = np.asarray(y_lower, dtype=float)
    y_upper = np.asarray(y_upper, dtype=float)
    coverage = float(np.mean((y_true >= y_lower) & (y_true <= y_upper)))
    avg_width = float(np.mean(y_upper - y_lower))
    alpha = 1.0 - coverage_target
    return {
        "coverage_achieved": coverage,
        "coverage_target": float(coverage_target),
        "coverage_gap": coverage - float(coverage_target),
        "avg_interval_width": avg_width,
        "winkler_score": winkler_score(y_true, y_lower, y_upper, alpha),
    }
