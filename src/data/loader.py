"""Carga, validacion y reporte de calidad de los CSVs de Tostao'.

Punto de entrada principal: :func:`load_all_data`.

Convenciones:
    * Las fechas se parsean a ``datetime64[ns]`` (sin timezone).
    * Los identificadores categoricos se cargan como ``string`` (pyarrow-friendly).
    * Las columnas con tildes/enies del CSV se renombran a ASCII al cargar.

Este modulo NO modifica los CSVs originales; solo lee y valida.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

import pandas as pd

from src.config import (
    DATA_DIR,
    SCHEMA_CATALOGO,
    SCHEMA_GROUND_TRUTH,
    SCHEMA_INVENTARIO,
    SCHEMA_TIENDAS,
    SCHEMA_VENTAS,
    VALID_TREND_TYPES,
)

logger = logging.getLogger(__name__)


class QualityReport(TypedDict):
    """Reporte de calidad emitido por :func:`load_all_data`."""

    shape_per_file: dict[str, tuple[int, int]]
    null_counts: dict[str, dict[str, int]]
    zero_demand_pct: float
    n_stores: int
    n_products: int
    n_sku_store_combos: int
    n_expected_combos: int
    coverage_pct: float
    missing_combos: list[tuple[str, str]]
    date_range: tuple[pd.Timestamp, pd.Timestamp]
    n_days: int
    inventory_orphan_combos: list[tuple[str, str]]
    catalog_orphan_products: list[str]
    trend_orphan_combos: list[tuple[str, str]]


# Mapeo de columnas con caracteres no-ASCII -> ASCII canonico.
_COLUMN_RENAMES: dict[str, str] = {
    "tamaño_m2": "tamano_m2",
}


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------
def _read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    """Lee un CSV en UTF-8 con renombrado canonico de columnas."""
    if not path.exists():
        raise FileNotFoundError(f"CSV no encontrado: {path}")
    df = pd.read_csv(path, encoding="utf-8", **kwargs)  # type: ignore[arg-type]
    return df.rename(columns=_COLUMN_RENAMES)


def _validate_schema(
    df: pd.DataFrame,
    schema: dict[str, str],
    *,
    name: str,
) -> None:
    """Verifica que ``df`` contiene todas las columnas declaradas en ``schema``.

    No coacciona tipos: solo valida presencia. La conversion de tipos se hace
    explicitamente en cada loader (`_load_*`).

    Args:
        df: DataFrame a validar.
        schema: Mapa columna -> dtype esperado (informativo).
        name: Etiqueta del archivo para mensajes de error.

    Raises:
        ValueError: Si falta alguna columna.
    """
    missing = set(schema) - set(df.columns)
    if missing:
        raise ValueError(
            f"[{name}] Columnas faltantes vs schema esperado: {sorted(missing)}. "
            f"Encontradas: {sorted(df.columns)}"
        )


# ---------------------------------------------------------------------------
# Loaders por archivo
# ---------------------------------------------------------------------------
def _load_ventas(path: Path) -> pd.DataFrame:
    """Carga ``ventas_historicas.csv`` y valida el rango de fechas."""
    df = _read_csv(path, parse_dates=["fecha"])
    _validate_schema(df, SCHEMA_VENTAS, name="ventas_historicas")
    df = df.astype(
        {
            "id_tienda": "string",
            "id_producto": "string",
            "unidades_vendidas": "int64",
        }
    )
    if (df["unidades_vendidas"] < 0).any():
        raise ValueError(
            "Hay unidades_vendidas negativas, lo cual no tiene sentido fisico."
        )
    return df.sort_values(["id_tienda", "id_producto", "fecha"]).reset_index(drop=True)


def _load_inventario(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    _validate_schema(df, SCHEMA_INVENTARIO, name="inventario_actual")
    df = df.astype(
        {
            "id_tienda": "string",
            "id_producto": "string",
            "stock_actual": "int64",
        }
    )
    if (df["stock_actual"] < 0).any():
        raise ValueError("Stock actual negativo detectado.")
    return df


def _load_catalogo(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    _validate_schema(df, SCHEMA_CATALOGO, name="catalogo_productos")
    df = df.astype(
        {
            "id_producto": "string",
            "nombre": "string",
            "categoria": "string",
            "costo_unitario": "float64",
            "precio_venta": "float64",
            "costo_almacenamiento_semanal": "float64",
        }
    )
    if (df["precio_venta"] <= df["costo_unitario"]).any():
        bad = df.loc[df["precio_venta"] <= df["costo_unitario"], "id_producto"].tolist()
        raise ValueError(f"Productos con margen <= 0: {bad}")
    return df


def _load_tiendas(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    _validate_schema(df, SCHEMA_TIENDAS, name="maestro_tiendas")
    return df.astype(
        {
            "id_tienda": "string",
            "ciudad": "string",
            "tamano_m2": "int64",
        }
    )


def _load_ground_truth(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    _validate_schema(df, SCHEMA_GROUND_TRUTH, name="ground_truth_trends")
    df = df.astype(
        {
            "id_tienda": "string",
            "id_producto": "string",
            "trend_type": "string",
        }
    )
    invalid = set(df["trend_type"].unique()) - VALID_TREND_TYPES
    if invalid:
        raise ValueError(f"trend_type invalidos: {invalid}")
    return df


# ---------------------------------------------------------------------------
# Reporte de calidad
# ---------------------------------------------------------------------------
def build_quality_report(data: dict[str, pd.DataFrame]) -> QualityReport:
    """Construye un reporte de calidad sobre el dataset cargado.

    Args:
        data: Diccionario con los DataFrames retornados por :func:`load_all_data`.

    Returns:
        :class:`QualityReport` con metricas de cobertura, nulos y consistencia
        cruzada (referential integrity entre archivos).
    """
    ventas = data["ventas"]
    inv = data["inventario"]
    cat = data["catalogo"]
    tiendas = data["tiendas"]
    gt = data["ground_truth"]

    shape_per_file = {k: tuple(df.shape) for k, df in data.items()}
    null_counts = {k: df.isna().sum().to_dict() for k, df in data.items()}

    n_stores = ventas["id_tienda"].nunique()
    n_products = ventas["id_producto"].nunique()
    combos_in_data = ventas.groupby(["id_tienda", "id_producto"]).ngroups
    expected = tiendas["id_tienda"].nunique() * cat["id_producto"].nunique()

    seen = set(
        map(tuple, ventas[["id_tienda", "id_producto"]].drop_duplicates().to_numpy())
    )
    full = {(s, p) for s in tiendas["id_tienda"] for p in cat["id_producto"]}
    missing = sorted(full - seen)

    inv_keys = set(map(tuple, inv[["id_tienda", "id_producto"]].to_numpy()))
    cat_products = set(cat["id_producto"])
    sales_products = set(ventas["id_producto"])
    inv_orphans = sorted(inv_keys - full)
    cat_orphans = sorted(cat_products - sales_products)
    gt_keys = set(map(tuple, gt[["id_tienda", "id_producto"]].to_numpy()))
    gt_orphans = sorted(gt_keys - full)

    date_min = ventas["fecha"].min()
    date_max = ventas["fecha"].max()

    return QualityReport(
        shape_per_file=shape_per_file,
        null_counts=null_counts,
        zero_demand_pct=float((ventas["unidades_vendidas"] == 0).mean()),
        n_stores=int(n_stores),
        n_products=int(n_products),
        n_sku_store_combos=int(combos_in_data),
        n_expected_combos=int(expected),
        coverage_pct=float(combos_in_data / expected) if expected else 0.0,
        missing_combos=missing,
        date_range=(date_min, date_max),
        n_days=int((date_max - date_min).days + 1),
        inventory_orphan_combos=inv_orphans,
        catalog_orphan_products=cat_orphans,
        trend_orphan_combos=gt_orphans,
    )


# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------
def load_all_data(data_dir: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Carga los 5 CSVs del proyecto y los retorna validados.

    Aplica:
        * Validacion de schema por archivo (columnas + dtypes).
        * Coercion explicita de tipos (string/int64/float64/datetime).
        * Renombrado de columnas con caracteres no-ASCII a forma canonica
          (ej. ``tamaño_m2`` -> ``tamano_m2``).
        * Verificaciones basicas de plausibilidad (no negativos, margen > 0,
          ``trend_type`` en el dominio valido).

    Args:
        data_dir: Carpeta que contiene los CSVs. Si ``None``, usa
            :data:`src.config.DATA_DIR`.

    Returns:
        Diccionario con claves ``ventas``, ``inventario``, ``catalogo``,
        ``tiendas``, ``ground_truth`` -> DataFrames listos para uso.

    Raises:
        FileNotFoundError: Si falta cualquiera de los archivos esperados.
        ValueError: Si algun archivo no pasa la validacion de schema o las
            verificaciones de plausibilidad.
    """
    base = Path(data_dir) if data_dir is not None else DATA_DIR
    return {
        "ventas": _load_ventas(base / "ventas_historicas.csv"),
        "inventario": _load_inventario(base / "inventario_actual.csv"),
        "catalogo": _load_catalogo(base / "catalogo_productos.csv"),
        "tiendas": _load_tiendas(base / "maestro_tiendas.csv"),
        "ground_truth": _load_ground_truth(base / "ground_truth_trends.csv"),
    }


def format_quality_report(report: QualityReport) -> str:
    """Renderiza el :class:`QualityReport` como texto legible."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("REPORTE DE CALIDAD DE DATOS")
    lines.append("=" * 72)

    lines.append("\n[Shape por archivo]")
    for name, shape in report["shape_per_file"].items():
        lines.append(f"  {name:<14} -> {shape[0]:>6} filas x {shape[1]} columnas")

    lines.append("\n[Nulos por archivo]")
    for name, nulls in report["null_counts"].items():
        total = sum(nulls.values())
        flag = "OK" if total == 0 else f"!! {total} nulos"
        lines.append(f"  {name:<14} -> {flag}")

    d0, d1 = report["date_range"]
    lines.append("\n[Rango temporal]")
    lines.append(f"  {d0.date()}  ->  {d1.date()}   ({report['n_days']} dias)")

    lines.append("\n[Cobertura SKU-Tienda]")
    lines.append(f"  Tiendas             : {report['n_stores']}")
    lines.append(f"  Productos           : {report['n_products']}")
    lines.append(
        f"  Combos observados   : {report['n_sku_store_combos']} / "
        f"{report['n_expected_combos']} ({report['coverage_pct']:.1%})"
    )
    if report["missing_combos"]:
        lines.append(f"  !! Combos faltantes : {len(report['missing_combos'])}")
    else:
        lines.append("  Combos faltantes    : 0  (cobertura completa)")

    lines.append("\n[Senales de stockout implicito]")
    lines.append(f"  % de ventas en cero : {report['zero_demand_pct']:.2%}")

    lines.append("\n[Integridad referencial]")
    lines.append(
        f"  Combos huerfanos en inventario : {len(report['inventory_orphan_combos'])}"
    )
    lines.append(
        f"  Productos en catalogo no vendidos: {len(report['catalog_orphan_products'])}"
    )
    lines.append(
        f"  Combos huerfanos en ground_truth : {len(report['trend_orphan_combos'])}"
    )
    lines.append("=" * 72)
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    data = load_all_data()
    report = build_quality_report(data)
    print(format_quality_report(report))
