"""Feature engineering para el forecaster global.

API publica:
    * :func:`flag_implicit_stockouts` -- marca ceros con vecinos positivos.
    * :func:`add_calendar_features` -- DOW, festivos colombianos, distancias.
    * :func:`add_lag_features` -- lags + rolling + EWM + slope local.
    * :func:`add_store_features` / :func:`add_product_features` / :func:`add_id_encodings`.
    * :func:`build_feature_matrix` -- pipeline orquestador train/test.
    * :func:`_propagate_lags_to_test` -- rellena lags del test usando el historial
      del train (implementacion completa, no stub).

Reglas
    * Los lags se calculan SOLO sobre el historial conocido. Para el test,
      se concatena (train_with_target + test_with_NaN_target_si_no_disponible)
      y luego se computa cada lag respetando el orden cronologico.
    * Los encoders categoricos se ajustan UNA SOLA VEZ sobre el universo
      conocido (`maestro_tiendas`, `catalogo`) para evitar inconsistencias
      entre folds del CV temporal.
    * `is_implicit_stockout` se marca pero NO se elimina del frame que
      alimenta los lags -- se elimina solo del target de entrenamiento, para
      que los lags reflejen la posicion calendarica correcta.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import holidays
import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    COUNTRY_CODE,
    EWM_SPAN,
    FEATURE_COLS,
    FEATURE_COLS_LAGS,
    HOLIDAY_HORIZON_DAYS,
    LAG_DAYS,
    LOCAL_TREND_WINDOW,
    ROLLING_CONFIGS,
    TARGET_COL,
)

GROUP_COLS: list[str] = ["id_tienda", "id_producto"]


# ---------------------------------------------------------------------------
# Stockouts implicitos
# ---------------------------------------------------------------------------
def flag_implicit_stockouts(df: pd.DataFrame) -> pd.DataFrame:
    """Marca ceros rodeados de demanda positiva como stockout probable.

    Anade la columna boolean ``is_implicit_stockout``. No filtra filas: el
    historial debe seguir intacto para que los lags queden alineados al
    calendario.

    Args:
        df: DataFrame con columnas ``id_tienda``, ``id_producto``, ``fecha``,
            ``unidades_vendidas``.

    Returns:
        El DataFrame ordenado por (tienda, producto, fecha) con la columna
        ``is_implicit_stockout`` agregada.
    """
    df = df.sort_values(GROUP_COLS + ["fecha"]).copy()
    grp = df.groupby(GROUP_COLS, observed=True)["unidades_vendidas"]
    lag1 = grp.shift(1)
    lead1 = grp.shift(-1)
    df["is_implicit_stockout"] = (
        (df["unidades_vendidas"] == 0) & (lag1 > 0) & (lead1 > 0)
    )
    return df


# ---------------------------------------------------------------------------
# Calendario y festivos
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HolidayContext:
    """Encapsula el calendario de festivos para un rango de anios."""

    country: str = COUNTRY_CODE

    def calendar(self, years: Iterable[int]) -> set[pd.Timestamp]:
        cal = holidays.country_holidays(self.country, years=list(years))
        return {pd.Timestamp(d) for d in cal.keys()}


def _required_years(dates: pd.Series) -> list[int]:
    """Anos cubiertos por la serie + 1 anio extra para el horizonte de festivos."""
    if dates.empty:
        return []
    y0 = int(pd.Timestamp(dates.min()).year)
    y1 = int(pd.Timestamp(dates.max()).year) + 1  # festivos del anio siguiente
    return list(range(y0, y1 + 1))


def _build_holiday_distance_maps(
    dates: pd.Series, *, ctx: HolidayContext
) -> tuple[dict[pd.Timestamp, int], dict[pd.Timestamp, int], set[pd.Timestamp]]:
    """Precomputa, por cada fecha unica, dias_al_proximo y desde_ultimo festivo.

    Mas eficiente que aplicar dos loops de 30 dias por fila: ordena las fechas
    unicas y barre el calendario una sola vez.
    """
    cal = ctx.calendar(_required_years(dates))
    unique_dates = sorted(pd.to_datetime(dates).unique())
    cal_sorted = sorted(cal)

    to_next: dict[pd.Timestamp, int] = {}
    from_last: dict[pd.Timestamp, int] = {}
    for d in unique_dates:
        d_ts = pd.Timestamp(d)
        # next
        nxt = HOLIDAY_HORIZON_DAYS
        for h in cal_sorted:
            if h > d_ts:
                delta = (h - d_ts).days
                if delta <= HOLIDAY_HORIZON_DAYS:
                    nxt = delta
                break
        # last
        prev = HOLIDAY_HORIZON_DAYS
        for h in reversed(cal_sorted):
            if h < d_ts:
                delta = (d_ts - h).days
                if delta <= HOLIDAY_HORIZON_DAYS:
                    prev = delta
                break
        to_next[d_ts] = nxt
        from_last[d_ts] = prev
    return to_next, from_last, cal


def add_calendar_features(
    df: pd.DataFrame, *, ctx: HolidayContext | None = None
) -> pd.DataFrame:
    """Anade features de calendario y festivos colombianos.

    Crea: ``day_of_week`` (0=Lun, 6=Dom), ``is_weekend`` (Vie-Dom),
    ``week_of_year``, ``month``, ``day_of_month``, ``is_holiday``,
    ``days_to_next_holiday``, ``days_from_last_holiday``.

    Args:
        df: DataFrame con columna ``fecha``.
        ctx: Contexto de festivos (default: Colombia). Inyectable para tests.

    Returns:
        Copia del DataFrame con las 8 features de calendario agregadas.
    """
    ctx = ctx or HolidayContext()
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])

    df["day_of_week"] = df["fecha"].dt.dayofweek.astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 4).astype("int8")  # Vie=4, Sab=5, Dom=6
    df["week_of_year"] = df["fecha"].dt.isocalendar().week.astype("int8")
    df["month"] = df["fecha"].dt.month.astype("int8")
    df["day_of_month"] = df["fecha"].dt.day.astype("int8")

    to_next, from_last, cal = _build_holiday_distance_maps(df["fecha"], ctx=ctx)
    df["is_holiday"] = df["fecha"].isin(cal).astype("int8")
    df["days_to_next_holiday"] = df["fecha"].map(to_next).astype("int16")
    df["days_from_last_holiday"] = df["fecha"].map(from_last).astype("int16")
    return df


# ---------------------------------------------------------------------------
# Lags / rolling / EWM / slope local
# ---------------------------------------------------------------------------
def _linear_slope(values: np.ndarray) -> float:
    """Pendiente de regresion lineal sobre una ventana 1D."""
    if len(values) < 4 or np.isnan(values).any():
        return 0.0
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def add_lag_features(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COL,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Calcula lags, rolling stats, EWM y slope local por SKU-Tienda.

    Importante:
        * Todos los rolling/EWM se calculan sobre ``target.shift(1)`` para
          NO incluir el dia actual (anti-leakage).
        * El slope local usa ventana de :data:`LOCAL_TREND_WINDOW` dias.
        * Las primeras filas por combo quedan con ``NaN`` -- el orquestador
          se encarga de descartarlas en el set de entrenamiento.

    Args:
        df: DataFrame ordenado o no, con ``fecha`` y ``target_col``.
        target_col: Columna sobre la que se calculan los lags.
        group_cols: Columnas que definen una serie. Default ``[id_tienda,
            id_producto]``.

    Returns:
        Copia con los features lag/rolling/ewm/slope agregados.
    """
    group_cols = group_cols or list(GROUP_COLS)
    df = df.sort_values(group_cols + ["fecha"]).copy()
    grp = df.groupby(group_cols, observed=True)[target_col]

    for k in LAG_DAYS:
        df[f"lag_{k}"] = grp.shift(k)

    shifted = grp.shift(1)  # serie shifted por 1 (no incluye dia actual)
    df_shifted = pd.DataFrame({"_s": shifted, **{c: df[c] for c in group_cols}})

    for window, stat, name in ROLLING_CONFIGS:
        if stat == "mean":
            df[name] = df_shifted.groupby(group_cols, observed=True)["_s"].transform(
                lambda x, w=window: x.rolling(w, min_periods=w).mean()
            )
        elif stat == "std":
            df[name] = df_shifted.groupby(group_cols, observed=True)["_s"].transform(
                lambda x, w=window: x.rolling(w, min_periods=w).std()
            )
        elif stat == "median":
            df[name] = df_shifted.groupby(group_cols, observed=True)["_s"].transform(
                lambda x, w=window: x.rolling(w, min_periods=w).median()
            )
        else:
            raise ValueError(f"Stat no soportada: {stat}")

    df[f"ewm_{EWM_SPAN}"] = df_shifted.groupby(group_cols, observed=True)[
        "_s"
    ].transform(lambda x: x.ewm(span=EWM_SPAN, min_periods=EWM_SPAN).mean())

    df["local_trend_slope"] = df_shifted.groupby(group_cols, observed=True)[
        "_s"
    ].transform(
        lambda x: x.rolling(LOCAL_TREND_WINDOW, min_periods=LOCAL_TREND_WINDOW).apply(
            _linear_slope, raw=True
        )
    )
    return df


# ---------------------------------------------------------------------------
# Tienda / producto / IDs
# ---------------------------------------------------------------------------
def _build_label_map(values: Iterable[str]) -> dict[str, int]:
    """Mapa determinista valor -> entero, ordenado por sorted unique."""
    return {v: i for i, v in enumerate(sorted(pd.unique(pd.Series(values).dropna())))}


def add_store_features(df: pd.DataFrame, maestro_tiendas: pd.DataFrame) -> pd.DataFrame:
    """Merge con atributos de tienda + encodings deterministas.

    Crea ``ciudad_encoded``, ``store_tier_encoded`` (qcut por tamano), y
    propaga ``tamano_m2``.
    """
    store_df = maestro_tiendas.copy()
    ciudad_map = _build_label_map(store_df["ciudad"])
    store_df["ciudad_encoded"] = store_df["ciudad"].map(ciudad_map).astype("int16")

    tier = pd.qcut(store_df["tamano_m2"], q=3, labels=["small", "medium", "large"])
    tier_map = {"small": 0, "medium": 1, "large": 2}
    store_df["store_tier_encoded"] = tier.astype(str).map(tier_map).astype("int8")

    cols = ["id_tienda", "tamano_m2", "ciudad_encoded", "store_tier_encoded"]
    return df.merge(store_df[cols], on="id_tienda", how="left")


def add_product_features(df: pd.DataFrame, catalogo: pd.DataFrame) -> pd.DataFrame:
    """Merge con atributos de producto + critical ratio.

    Crea ``categoria_encoded``, ``precio_venta``, ``margen``, ``critical_ratio``.
    """
    cat_df = catalogo.copy()
    cat_map = _build_label_map(cat_df["categoria"])
    cat_df["categoria_encoded"] = cat_df["categoria"].map(cat_map).astype("int8")
    cat_df["margen"] = cat_df["precio_venta"] - cat_df["costo_unitario"]
    cat_df["critical_ratio"] = cat_df["margen"] / (
        cat_df["margen"] + cat_df["costo_almacenamiento_semanal"]
    )

    cols = [
        "id_producto",
        "categoria_encoded",
        "precio_venta",
        "margen",
        "critical_ratio",
    ]
    return df.merge(cat_df[cols], on="id_producto", how="left")


def add_id_encodings(
    df: pd.DataFrame,
    *,
    maestro_tiendas: pd.DataFrame,
    catalogo: pd.DataFrame,
) -> pd.DataFrame:
    """Label encoding determinista de ``id_tienda`` e ``id_producto``.

    Usa el universo conocido (maestro y catalogo) como referencia para que el
    encoding sea consistente entre folds del CV.
    """
    df = df.copy()
    store_map = _build_label_map(maestro_tiendas["id_tienda"])
    prod_map = _build_label_map(catalogo["id_producto"])
    df["id_tienda_encoded"] = df["id_tienda"].map(store_map).astype("int16")
    df["id_producto_encoded"] = df["id_producto"].map(prod_map).astype("int16")
    return df


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def _propagate_lags_to_test(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    """Calcula los lags del test set usando el historial conocido del train.

    Estrategia (direct multi-output, no recursiva):
        1. Concatenamos train + test en una sola tabla, conservando el target
           del train (incluso para filas marcadas como stockout, ver `flag_*`)
           y dejando ``NaN`` en el target de las filas de test que no lo
           tienen (caso de inferencia real).
        2. Si el caller proporciona el target en test_df (ej. validacion contra
           ground truth), se respeta -- los lags subsiguientes podran
           encadenarse.
        3. Recomputamos lags/rolling/EWM/slope sobre la tabla combinada
           ordenada cronologicamente. Cada feature en una fila de test mira
           SOLO hacia el pasado calendarico, asi que no hay leakage.
        4. Retornamos las filas de test con sus features lag rellenas.

    En forecasting puro (sin target en test), tendremos:
        - ``lag_1`` definido solo para el primer dia de test.
        - ``lag_7``/``lag_14`` definidos para todas las fechas si el horizonte
          de test <= lag.
        - rolling/EWM caen a NaN tras varias filas si no hay target conocido.
          El forecaster puede sobreescribirlos en modo recursivo cuando
          predice cada paso.

    Args:
        train_df: DataFrame con features + target conocidos.
        test_df: DataFrame con metadata (fecha, ids, opcionalmente target).

    Returns:
        Copia de ``test_df`` con las columnas de :data:`FEATURE_COLS_LAGS`
        rellenas a partir del historial.
    """
    if TARGET_COL not in test_df.columns:
        test_df = test_df.copy()
        test_df[TARGET_COL] = np.nan

    train_marker = train_df[GROUP_COLS + ["fecha", TARGET_COL]].assign(_split="train")
    test_marker = test_df[GROUP_COLS + ["fecha", TARGET_COL]].assign(_split="test")
    combined = pd.concat([train_marker, test_marker], ignore_index=True)
    combined = add_lag_features(combined, target_col=TARGET_COL)

    lag_cols = list(FEATURE_COLS_LAGS)
    test_lags = combined.loc[
        combined["_split"] == "test", GROUP_COLS + ["fecha"] + lag_cols
    ]

    enriched = test_df.drop(columns=[c for c in lag_cols if c in test_df.columns])
    enriched = enriched.merge(test_lags, on=GROUP_COLS + ["fecha"], how="left")
    return enriched


def build_feature_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    maestro_tiendas: pd.DataFrame,
    catalogo: pd.DataFrame,
    drop_train_stockouts: bool = True,
    drop_lag_nans: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pipeline completo: produce X_train (con target) y X_test (sin target).

    Steps:
        1. (train) `flag_implicit_stockouts`. Si ``drop_train_stockouts``,
           se filtran del *target* de entrenamiento DESPUES de calcular lags.
        2. Calendario, store, producto, ID encoding -- iguales en train y test.
        3. ``add_lag_features`` sobre train.
        4. ``_propagate_lags_to_test`` sobre test.
        5. Si ``drop_lag_nans``, eliminamos del train las filas con NaN en
           cualquier feature de lag (las primeras 14 obs por SKU-Tienda).

    Args:
        train_df: DataFrame con ``fecha``, ``id_tienda``, ``id_producto``,
            ``unidades_vendidas``.
        test_df: Mismo schema; el target puede estar ausente para inferencia.
        maestro_tiendas: Catalogo de tiendas (universo para encoding).
        catalogo: Catalogo de productos (universo para encoding).
        drop_train_stockouts: Si True, filas con ``is_implicit_stockout==True``
            se quitan del train final (no se quitan del frame que alimenta los
            lags, solo del target).
        drop_lag_nans: Si True, drop filas con NaN en columnas de lag.

    Returns:
        ``(X_train, X_test)`` con las columnas en :data:`FEATURE_COLS` mas el
        target en X_train (sin target en X_test).
    """
    train_df = flag_implicit_stockouts(train_df)
    if "is_implicit_stockout" not in test_df.columns:
        test_df = test_df.copy()
        test_df["is_implicit_stockout"] = False

    def _common_pipe(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.pipe(add_calendar_features)
            .pipe(add_store_features, maestro_tiendas=maestro_tiendas)
            .pipe(add_product_features, catalogo=catalogo)
            .pipe(add_id_encodings, maestro_tiendas=maestro_tiendas, catalogo=catalogo)
        )

    train_full = _common_pipe(train_df)
    test_full = _common_pipe(test_df)

    train_full = add_lag_features(train_full)
    test_full = _propagate_lags_to_test(train_full, test_full)

    if drop_train_stockouts:
        train_full = train_full.loc[~train_full["is_implicit_stockout"]].copy()

    if drop_lag_nans:
        train_full = train_full.dropna(subset=list(FEATURE_COLS_LAGS))

    feature_cols = list(FEATURE_COLS)
    keep_meta = ["fecha", "id_tienda", "id_producto"]

    X_train = train_full[keep_meta + feature_cols + [TARGET_COL]].reset_index(drop=True)
    X_test = test_full[keep_meta + feature_cols].reset_index(drop=True)
    return X_train, X_test


def get_categorical_feature_indices(
    feature_cols: Iterable[str] = FEATURE_COLS,
) -> list[int]:
    """Indices de columnas categoricas en ``feature_cols`` para LightGBM."""
    cols = list(feature_cols)
    return [cols.index(c) for c in CATEGORICAL_FEATURES if c in cols]
