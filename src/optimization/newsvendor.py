"""Optimizador Newsvendor: convierte el pronostico + intervalos conformales
en una decision de pedido por SKU-Tienda.

Conexion teorica central del proyecto
-------------------------------------
La cantidad optima de pedido del modelo Newsvendor con costos
``Cu = precio - costo_unitario`` (margen perdido por stockout) y
``Co = costo_almacenamiento_semanal`` (costo de overstock) es:

    Q* = F^{-1}(CR)        donde   CR = Cu / (Cu + Co)

Cuando la incertidumbre se cuantifica con un intervalo conformal one-sided
calibrado al nivel ``CR`` (coverage_target = critical_ratio), el upper bound
del intervalo **es directamente** ``F^{-1}(CR)``. Por lo tanto:

    Q* = y_upper                (sin recalcular cuantiles parametricos)

El optimizador no asume Normal: el upper conformal absorbe la forma real de
la distribucion via los residuos asimetricos del modelo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


# ===========================================================================
# Tipos
# ===========================================================================
@dataclass(frozen=True)
class OrderDecision:
    """Decision de pedido para un SKU-Tienda (representacion in-memory)."""

    id_tienda: str
    id_producto: str
    forecast_media_semanal: float
    cantidad_target_q_star: float
    stock_actual: int
    critical_ratio: float
    cantidad_pedido: int
    alerta: str = ""
    costo_total_esperado: float = 0.0

    @property
    def cobertura_dias(self) -> float:
        """Dias de cobertura del stock si no se pidiera nada mas."""
        daily_mean = self.forecast_media_semanal / 7.0
        return self.stock_actual / daily_mean if daily_mean > 0 else float("inf")


# ===========================================================================
# Optimizador
# ===========================================================================
class NewsvendorOptimizer:
    """Convierte predicciones + intervalos conformales + inventario en pedidos.

    Args:
        catalogo: DataFrame con columnas ``id_producto, costo_unitario,
            precio_venta, costo_almacenamiento_semanal``. El optimizador
            calcula ``cu``, ``co`` y ``critical_ratio`` internamente.
    """

    REQUIRED_CATALOG_COLS = {
        "id_producto",
        "costo_unitario",
        "precio_venta",
        "costo_almacenamiento_semanal",
    }
    REQUIRED_PRED_COLS = {"id_tienda", "id_producto", "y_pred", "y_upper"}

    def __init__(self, catalogo: pd.DataFrame) -> None:
        missing = self.REQUIRED_CATALOG_COLS - set(catalogo.columns)
        if missing:
            raise ValueError(f"catalogo no tiene columnas {missing}")
        self.catalogo = catalogo.copy()
        self._compute_cost_parameters()

    def _compute_cost_parameters(self) -> None:
        cat = self.catalogo
        cat["cu"] = cat["precio_venta"] - cat["costo_unitario"]
        cat["co"] = cat["costo_almacenamiento_semanal"]
        if (cat["cu"] <= 0).any():
            bad = cat.loc[cat["cu"] <= 0, "id_producto"].tolist()
            raise ValueError(f"Productos con margen <= 0: {bad}")
        cat["critical_ratio"] = cat["cu"] / (cat["cu"] + cat["co"])

    # ------------------------------------------------------------------
    def compute_order(
        self,
        predictions: pd.DataFrame,
        inventario: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calcula la cantidad optima de pedido por SKU-Tienda.

        Reglas:
            * ``q_star = y_upper`` (upper bound conformal one-sided al nivel CR).
            * ``cantidad_pedido = max(0, ceil(q_star) - stock_actual)``.
            * ``alerta = "SOBRESTOCK"`` cuando ``stock_actual > q_star``.
            * ``costo_total_esperado`` proxy puntual usando la media puntual del
              forecast (penaliza la diferencia entre Q* y la media), util para
              priorizar SKU-Tiendas con mayor riesgo agregado.

        Args:
            predictions: DataFrame con columnas
                ``id_tienda``, ``id_producto``, ``y_pred`` (media semanal),
                ``y_upper`` (upper conformal = Q*). Opcional:
                ``coverage_target`` (se valida si esta).
            inventario: DataFrame con ``id_tienda``, ``id_producto``,
                ``stock_actual``.

        Returns:
            DataFrame con la recomendacion de pedido por SKU-Tienda y
            schema canonico de ``outputs/purchase_orders.csv``.
        """
        missing = self.REQUIRED_PRED_COLS - set(predictions.columns)
        if missing:
            raise ValueError(f"predictions no tiene columnas {missing}")

        df = predictions.merge(
            inventario[["id_tienda", "id_producto", "stock_actual"]],
            on=["id_tienda", "id_producto"],
            how="left",
        ).merge(
            self.catalogo[["id_producto", "cu", "co", "critical_ratio"]],
            on="id_producto",
            how="left",
        )
        if df["stock_actual"].isna().any():
            missing_combos = (
                df.loc[df["stock_actual"].isna(), ["id_tienda", "id_producto"]]
                .head()
                .to_dict("records")
            )
            raise ValueError(
                f"Sin inventario para algunos combos. Ejemplo: {missing_combos}"
            )

        if "coverage_target" in df.columns:
            mismatch = (df["coverage_target"] - df["critical_ratio"]).abs()
            if (mismatch > 1e-3).any():
                logger.warning(
                    "coverage_target del forecaster no coincide con critical_ratio "
                    "del catalogo en %d filas. Max diff: %.4f",
                    int((mismatch > 1e-3).sum()),
                    float(mismatch.max()),
                )

        df["q_star"] = df["y_upper"].astype(float)
        df["cantidad_pedido"] = (
            (df["q_star"] - df["stock_actual"]).clip(lower=0).apply(np.ceil).astype(int)
        )
        df["alerta"] = np.where(
            df["stock_actual"] > df["q_star"], "SOBRESTOCK: stock supera Q*", ""
        )
        # Costo puntual esperado proxy (usando la media; el costo verdadero se
        # calcula sobre la realizacion via compute_business_impact).
        df["costo_esperado_stockout"] = df["cu"] * np.maximum(
            0.0, df["y_pred"] - df["q_star"]
        )
        df["costo_esperado_overstock"] = df["co"] * np.maximum(
            0.0, df["q_star"] - df["y_pred"]
        )
        df["costo_total_esperado"] = (
            df["costo_esperado_stockout"] + df["costo_esperado_overstock"]
        )

        out = df[
            [
                "id_tienda",
                "id_producto",
                "y_pred",
                "y_upper",
                "stock_actual",
                "critical_ratio",
                "cantidad_pedido",
                "alerta",
                "costo_total_esperado",
            ]
        ].rename(
            columns={
                "y_pred": "forecast_media_semanal",
                "y_upper": "cantidad_target_q_star",
            }
        )
        return out.sort_values(["id_tienda", "id_producto"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    def compute_expected_cost(
        self,
        Q: float,
        mu: float,
        sigma: float,
        cu: float,
        co: float,
    ) -> float:
        """Costo total esperado para un pedido ``Q`` bajo Normal(mu, sigma).

        Formula cerrada del Newsvendor con demanda Normal:

            E[max(D-Q, 0)] = (mu - Q)·(1 - Phi(z)) + sigma·phi(z)
            E[max(Q-D, 0)] = (Q - mu)·Phi(z)        + sigma·phi(z)

        donde ``z = (Q - mu)/sigma`` y ``phi``, ``Phi`` son la pdf y cdf de
        la Normal estandar. El costo total = ``cu * E[stockout] + co * E[overstock]``.

        Args:
            Q: Cantidad pedida.
            mu: Demanda esperada (media del forecast).
            sigma: Desviacion estandar del forecast.
            cu: Costo de underage (stockout).
            co: Costo de overage (overstock).

        Returns:
            Costo total esperado (float >= 0).
        """
        if sigma <= 0:
            # Demanda determinista: costo solo si Q != mu
            if Q < mu:
                return cu * (mu - Q)
            return co * (Q - mu)
        z = (Q - mu) / sigma
        phi = norm.pdf(z)
        Phi = norm.cdf(z)
        loss_stockout = (mu - Q) * (1.0 - Phi) + sigma * phi
        loss_overstock = (Q - mu) * Phi + sigma * phi
        return float(cu * loss_stockout + co * loss_overstock)

    # ------------------------------------------------------------------
    def sensitivity_analysis(
        self,
        id_tienda: str,
        id_producto: str,
        forecast_mean: float,
        forecast_std: float,
        n_points: int = 60,
    ) -> pd.DataFrame:
        """Curva de costo total vs Q (analisis de sensibilidad).

        Args:
            id_tienda: ID de la tienda (informativo, queda en el output).
            id_producto: ID del producto (debe existir en el catalogo).
            forecast_mean: Demanda esperada para la ventana.
            forecast_std: Desviacion estandar del forecast (dispersion del modelo).
            n_points: Numero de puntos en la curva.

        Returns:
            DataFrame con ``Q``, ``costo_stockout``, ``costo_overstock``,
            ``costo_total``, ``Q_optimo_normal`` (cuantil CR de la Normal).
        """
        prod = self.catalogo.query("id_producto == @id_producto")
        if prod.empty:
            raise KeyError(f"id_producto {id_producto!r} no esta en el catalogo.")
        cu = float(prod["cu"].iloc[0])
        co = float(prod["co"].iloc[0])
        cr = float(prod["critical_ratio"].iloc[0])

        sigma = max(float(forecast_std), 1e-6)
        q_lo = max(0.0, forecast_mean - 4.0 * sigma)
        q_hi = forecast_mean + 5.0 * sigma  # extiende a la derecha porque CR ~ 0.99
        Q_range = np.linspace(q_lo, q_hi, n_points)

        rows = []
        for Q in Q_range:
            z = (Q - forecast_mean) / sigma
            phi = norm.pdf(z)
            Phi = norm.cdf(z)
            cost_so = cu * ((forecast_mean - Q) * (1.0 - Phi) + sigma * phi)
            cost_os = co * ((Q - forecast_mean) * Phi + sigma * phi)
            rows.append(
                {
                    "Q": float(Q),
                    "costo_stockout": float(cost_so),
                    "costo_overstock": float(cost_os),
                    "costo_total": float(cost_so + cost_os),
                }
            )
        df = pd.DataFrame(rows)
        df["id_tienda"] = id_tienda
        df["id_producto"] = id_producto
        df["Q_optimo_normal"] = float(norm.ppf(cr, loc=forecast_mean, scale=sigma))
        return df


# ===========================================================================
# Impacto de negocio
# ===========================================================================
def compute_business_impact(
    orders_df: pd.DataFrame,
    actual_demand: pd.DataFrame,
    catalogo: pd.DataFrame,
    *,
    demand_col: str = "demanda_real",
) -> dict:
    """Compara costo realizado del modelo vs baseline naive (pedir la media).

    Para cada SKU-Tienda calcula:
        * Pedido recomendado por el modelo: ``cantidad_pedido``.
        * Pedido baseline naive: ``max(0, forecast_media_semanal - stock_actual)``
          (pedir lo que falta para llegar a la media historica).
        * Disponibilidad de la tienda en la semana objetivo:
          ``disponibilidad = stock_actual + cantidad_pedido``.
          El stock_actual ya cubre demanda y debe sumarse al pedido al
          calcular costo realizado; si solo se compara ``pedido vs demanda``,
          se infla artificialmente el stockout cuando hay stock disponible
          (y el overstock cuando no se pide nada).
        * Costo realizado bajo demanda observada:
          ``cu * max(0, demanda - disponibilidad) + co * max(0, disponibilidad - demanda)``.

    Reporta:
        ``costo_total_modelo_COP``, ``costo_total_naive_COP``, ``ahorro_COP``,
        ``ahorro_pct``, decomposicion ``ahorro_*_stockout`` / ``ahorro_*_overstock``,
        proporciones de fallos por modo.

    Args:
        orders_df: Output de :meth:`NewsvendorOptimizer.compute_order`.
        actual_demand: DataFrame con ``id_tienda``, ``id_producto``, y la
            columna ``demanda_real`` (suma observada en la semana objetivo).
        catalogo: Catalogo con ``cu`` y ``co`` (o ``costo_unitario``,
            ``precio_venta``, ``costo_almacenamiento_semanal``).

    Returns:
        Dict con todas las metricas de impacto.
    """
    cat = catalogo.copy()
    if "cu" not in cat.columns:
        cat["cu"] = cat["precio_venta"] - cat["costo_unitario"]
    if "co" not in cat.columns:
        cat["co"] = cat["costo_almacenamiento_semanal"]

    df = orders_df.merge(
        actual_demand[["id_tienda", "id_producto", demand_col]],
        on=["id_tienda", "id_producto"],
        how="inner",
    ).merge(cat[["id_producto", "cu", "co"]], on="id_producto", how="left")

    if df.empty:
        raise ValueError("Sin overlap entre orders y actual_demand.")

    # Disponibilidad del modelo: lo que la tienda tiene para vender en la
    # semana objetivo = inventario inicial + pedido recibido. Esta es la
    # cantidad correcta a comparar contra la demanda realizada.
    df["disponibilidad_modelo"] = df["stock_actual"] + df["cantidad_pedido"]
    df["stockout_modelo"] = (
        np.maximum(0.0, df[demand_col] - df["disponibilidad_modelo"]) * df["cu"]
    )
    df["overstock_modelo"] = (
        np.maximum(0.0, df["disponibilidad_modelo"] - df[demand_col]) * df["co"]
    )
    df["costo_modelo"] = df["stockout_modelo"] + df["overstock_modelo"]

    # Baseline naive: pedir lo que falta para llegar a la media historica.
    df["pedido_naive"] = (
        np.maximum(0.0, df["forecast_media_semanal"] - df["stock_actual"])
        .apply(np.ceil)
        .astype(int)
    )
    df["disponibilidad_naive"] = df["stock_actual"] + df["pedido_naive"]
    df["stockout_naive"] = (
        np.maximum(0.0, df[demand_col] - df["disponibilidad_naive"]) * df["cu"]
    )
    df["overstock_naive"] = (
        np.maximum(0.0, df["disponibilidad_naive"] - df[demand_col]) * df["co"]
    )
    df["costo_naive"] = df["stockout_naive"] + df["overstock_naive"]

    total_modelo = float(df["costo_modelo"].sum())
    total_naive = float(df["costo_naive"].sum())
    ahorro = total_naive - total_modelo
    ahorro_pct = (ahorro / total_naive * 100.0) if total_naive > 0 else float("nan")

    # Decomposicion del ahorro
    ahorro_stockout = float((df["stockout_naive"] - df["stockout_modelo"]).sum())
    ahorro_overstock = float((df["overstock_naive"] - df["overstock_modelo"]).sum())

    return {
        "n_sku_tienda_evaluados": int(len(df)),
        "demanda_real_total": float(df[demand_col].sum()),
        "pedido_modelo_total": int(df["cantidad_pedido"].sum()),
        "pedido_naive_total": int(df["pedido_naive"].sum()),
        "costo_total_modelo_COP": round(total_modelo),
        "costo_total_naive_COP": round(total_naive),
        "ahorro_COP": round(ahorro),
        "ahorro_pct": round(ahorro_pct, 2),
        "ahorro_via_menos_stockout_COP": round(ahorro_stockout),
        "ahorro_via_menos_overstock_COP": round(ahorro_overstock),
        "pct_combos_con_stockout_modelo": float((df["stockout_modelo"] > 0).mean()),
        "pct_combos_con_overstock_modelo": float((df["overstock_modelo"] > 0).mean()),
        "pct_combos_con_stockout_naive": float((df["stockout_naive"] > 0).mean()),
        "pct_combos_con_overstock_naive": float((df["overstock_naive"] > 0).mean()),
        "_detail_df": df,
    }
