"""Modelo LightGBM global multi-serie para pronostico de demanda.

Una sola instancia de :class:`GlobalDemandForecaster` aprende sobre las 160
series SKU-Tienda gracias al label-encoding de IDs como features. La clase
soporta ``objective='regression'`` (media) y ``objective='quantile'`` (cuantil
arbitrario), early stopping opcional y predicciones forzadas a ser >= 0.

Decisiones del skill:
    * ``num_leaves=31`` y ``min_child_samples=20`` para evitar overfit con 91 dias.
    * Predicciones siempre >= 0: la demanda no puede ser negativa.
    * Early stopping en un set de validacion separado del calibration set
      del CP (se pasa explicitamente en `fit`).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)


class GlobalDemandForecaster:
    """LightGBM global con soporte de regression + quantile + early stopping.

    Args:
        objective: ``'regression'`` para la media o ``'quantile'`` para el
            cuantil definido por ``alpha``.
        alpha: Cuantil objetivo cuando ``objective='quantile'``. Ignorado para
            regression. Default 0.9.
        random_state: Semilla.
        **lgbm_kwargs: Override de cualquier parametro base de LightGBM.

    Attributes:
        model: La instancia subyacente de :class:`LGBMRegressor`.
    """

    LGBM_BASE_PARAMS: dict[str, Any] = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "subsample_freq": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_jobs": -1,
        "verbose": -1,
    }

    def __init__(
        self,
        objective: str = "regression",
        alpha: float = 0.9,
        random_state: int = RANDOM_SEED,
        **lgbm_kwargs: Any,
    ) -> None:
        if objective not in {"regression", "quantile"}:
            raise ValueError(
                f"objective debe ser 'regression' o 'quantile', no {objective!r}"
            )
        self.objective = objective
        self.alpha = alpha

        params = {**self.LGBM_BASE_PARAMS, "random_state": random_state, **lgbm_kwargs}
        if objective == "quantile":
            params["objective"] = "quantile"
            params["alpha"] = alpha
        else:
            params["objective"] = "regression"

        self.model = LGBMRegressor(**params)
        self._feature_cols: Optional[list[str]] = None
        self._categorical_indices: list[int] | None = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
        feature_cols: Optional[list[str]] = None,
        categorical_features: Optional[list[str]] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = False,
    ) -> "GlobalDemandForecaster":
        """Ajusta el modelo. Usa early stopping si se proporciona ``X_val``/``y_val``.

        Args:
            X_train: DataFrame de features para entrenar.
            y_train: Target alineado con ``X_train``.
            X_val: Opcional. Set de validacion para early stopping (NO mezclar
                con el calibration set del CP).
            y_val: Target del set de validacion.
            feature_cols: Lista explicita de features. Si ``None``, usa todas
                las columnas de ``X_train``.
            categorical_features: Nombres de features categoricas (label-encoded
                como enteros) que LightGBM tratara como categoricas.
            early_stopping_rounds: Rondas sin mejora antes de parar.
            verbose: Imprimir progreso de entrenamiento.

        Returns:
            self (encadenable).
        """
        self._feature_cols = (
            list(feature_cols) if feature_cols else list(X_train.columns)
        )
        cat_idx: list[int] = []
        if categorical_features:
            cat_idx = [
                self._feature_cols.index(c)
                for c in categorical_features
                if c in self._feature_cols
            ]
        self._categorical_indices = cat_idx

        X_tr = X_train[self._feature_cols].to_numpy()
        y_tr = np.asarray(y_train)

        callbacks: list[Any] = []
        fit_kwargs: dict[str, Any] = {}

        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [
                (X_val[self._feature_cols].to_numpy(), np.asarray(y_val))
            ]
            fit_kwargs["eval_metric"] = (
                "l1" if self.objective == "regression" else "quantile"
            )
            callbacks.append(
                early_stopping(stopping_rounds=early_stopping_rounds, verbose=verbose)
            )
        if verbose:
            callbacks.append(log_evaluation(period=100))
        if callbacks:
            fit_kwargs["callbacks"] = callbacks
        if cat_idx:
            fit_kwargs["categorical_feature"] = cat_idx

        self.model.fit(X_tr, y_tr, **fit_kwargs)
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predice demanda; recorta a >= 0."""
        self._assert_fitted()
        Xn = X[self._feature_cols].to_numpy()
        preds = self.model.predict(Xn)
        return np.maximum(0.0, np.asarray(preds, dtype=float))

    # ------------------------------------------------------------------
    def get_feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """Retorna :class:`pd.Series` de feature importance ordenada descendente.

        Args:
            importance_type: ``'split'`` (default LGBM, # de splits) o
                ``'gain'`` (suma de ganancias). Default ``'gain'`` por mayor
                interpretabilidad de negocio.
        """
        self._assert_fitted()
        booster = self.model.booster_
        importances = booster.feature_importance(importance_type=importance_type)
        return pd.Series(
            importances, index=self._feature_cols, name=f"importance_{importance_type}"
        ).sort_values(ascending=False)

    # ------------------------------------------------------------------
    @property
    def best_iteration(self) -> int | None:
        """Retorna ``best_iteration_`` si hubo early stopping, sino ``None``."""
        self._assert_fitted()
        return getattr(self.model, "best_iteration_", None)

    @property
    def feature_cols(self) -> list[str]:
        self._assert_fitted()
        assert self._feature_cols is not None
        return list(self._feature_cols)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _assert_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Forecaster no ajustado. Llamar .fit() primero.")
