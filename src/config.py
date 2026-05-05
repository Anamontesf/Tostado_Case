"""Constantes globales del proyecto Tostao' Demand Forecasting.

Centraliza parametros del negocio y rutas. NUNCA hardcodear estos valores
dentro de los modulos de datos, modelos u optimizacion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "outputs"
NOTEBOOK_DIR: Final[Path] = PROJECT_ROOT / "notebooks"

# ---------------------------------------------------------------------------
# Horizonte y validacion
# ---------------------------------------------------------------------------
FORECAST_HORIZON_DAYS: Final[int] = 7
CALIBRATION_WEEKS: Final[int] = 3
MIN_TRAIN_WEEKS: Final[int] = 8

# ---------------------------------------------------------------------------
# Conformal Prediction
# ---------------------------------------------------------------------------
# Alpha se calcula POR SKU desde el critical ratio (alpha = 1 - CR).
# Mantener None aqui obliga al consumidor a derivarlo en runtime.
CONFORMAL_ALPHA: Final[float | None] = None

# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
RANDOM_SEED: Final[int] = 42

# ---------------------------------------------------------------------------
# Negocio (Colombia)
# ---------------------------------------------------------------------------
COUNTRY_CODE: Final[str] = "CO"

# Esquemas esperados por archivo (columna -> dtype objetivo).
# El loader valida estos schemas; cambios aqui obligan a regenerar fixtures.
SCHEMA_VENTAS: Final[dict[str, str]] = {
    "fecha": "datetime64[ns]",
    "id_tienda": "string",
    "id_producto": "string",
    "unidades_vendidas": "int64",
}
SCHEMA_INVENTARIO: Final[dict[str, str]] = {
    "id_tienda": "string",
    "id_producto": "string",
    "stock_actual": "int64",
}
SCHEMA_CATALOGO: Final[dict[str, str]] = {
    "id_producto": "string",
    "nombre": "string",
    "categoria": "string",
    "costo_unitario": "float64",
    "precio_venta": "float64",
    "costo_almacenamiento_semanal": "float64",
}
SCHEMA_TIENDAS: Final[dict[str, str]] = {
    "id_tienda": "string",
    "ciudad": "string",
    "tamano_m2": "int64",
}
SCHEMA_GROUND_TRUTH: Final[dict[str, str]] = {
    "id_tienda": "string",
    "id_producto": "string",
    "trend_type": "string",
}

VALID_TREND_TYPES: Final[frozenset[str]] = frozenset(
    {"up", "down", "seasonal", "random"}
)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
TARGET_COL: Final[str] = "unidades_vendidas"
GROUP_COLS: Final[tuple[str, ...]] = ("id_tienda", "id_producto")

LAG_DAYS: Final[tuple[int, ...]] = (1, 7, 14)

# (window, stat, name) -- stats validas: mean, std, median.
ROLLING_CONFIGS: Final[tuple[tuple[int, str, str], ...]] = (
    (7, "mean", "rolling_mean_7"),
    (7, "std", "rolling_std_7"),
    (14, "mean", "rolling_mean_14"),
    (7, "median", "rolling_median_7"),
)

EWM_SPAN: Final[int] = 7
LOCAL_TREND_WINDOW: Final[int] = 14
HOLIDAY_HORIZON_DAYS: Final[int] = 30  # cap para days_to/from_holiday

# Numero minimo de dias historicos requerido por combo SKU-Tienda para calcular
# el feature mas largo (rolling_mean_14 con shift(1)).
MIN_HISTORY_DAYS: Final[int] = max(LAG_DAYS) + 1  # = 15

FEATURE_COLS_CALENDAR: Final[tuple[str, ...]] = (
    "day_of_week",
    "is_weekend",
    "week_of_year",
    "month",
    "day_of_month",
    "is_holiday",
    "days_to_next_holiday",
    "days_from_last_holiday",
)

FEATURE_COLS_LAGS: Final[tuple[str, ...]] = tuple(
    [f"lag_{k}" for k in LAG_DAYS]
    + [name for _, _, name in ROLLING_CONFIGS]
    + [f"ewm_{EWM_SPAN}", "local_trend_slope"]
)

FEATURE_COLS_STORE: Final[tuple[str, ...]] = (
    "tamano_m2",
    "ciudad_encoded",
    "store_tier_encoded",
)

FEATURE_COLS_PRODUCT: Final[tuple[str, ...]] = (
    "categoria_encoded",
    "precio_venta",
    "margen",
    "critical_ratio",
)

FEATURE_COLS_IDS: Final[tuple[str, ...]] = (
    "id_tienda_encoded",
    "id_producto_encoded",
)

FEATURE_COLS: Final[tuple[str, ...]] = (
    FEATURE_COLS_CALENDAR
    + FEATURE_COLS_LAGS
    + FEATURE_COLS_STORE
    + FEATURE_COLS_PRODUCT
    + FEATURE_COLS_IDS
)

# Columnas que LightGBM debe tratar como categoricas (label-encoded enteros).
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "ciudad_encoded",
    "store_tier_encoded",
    "categoria_encoded",
    "id_tienda_encoded",
    "id_producto_encoded",
)
