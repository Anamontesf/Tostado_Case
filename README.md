# Tostado_Case
Sistema end-to-end que pronostica demanda semanal por SKU-Tienda y produce la cantidad óptima de pedido para minimizar el costo total esperado (stockout + overstock). Combina un LightGBM sobre las 160 series con Conformal Prediction calibrado al Critical Ratio del Newsvendor

## Instalación

Requiere Python 3.11.

```bash
# Crear entorno virtual e instalar dependencias
uv venv --python 3.11 .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate            # Windows
uv pip install -r requirements.txt
```

`pip + venv` también funciona si no se dispone de `uv`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

```bash
# Pipeline completo (regenera outputs/predictions_week_next.csv y outputs/purchase_orders.csv)
python -m src.main

# Tests
pytest tests/ -v

# Notebooks (ejecutar en orden)
jupyter notebook notebooks/
```

Atajos vía `Makefile`:

```bash
make install     # crea venv e instala dependencias
make predict     # equivalente a python -m src.main
make test        # pytest -v
make all         # clean + test + predict + ejecuta los 4 notebooks
make clean       # borra outputs/ y caches
```

## Arquitectura

Pipeline en cinco etapas, ejecutado end-to-end por `python -m src.main`:

```
data/ (5 CSVs)
   │
   ▼
src/data/loader.py            ── carga + validación de schemas
   │
   ▼
src/data/preprocessor.py      ── 26 features (calendario, lag, rolling/EWM,
   │                             slope, IDs, atributos producto/tienda)
   ▼
src/models/forecaster.py      ── LightGBM global multi-serie (160 series)
   │
   ▼
src/models/conformal.py       ── Conformal Prediction unilateral, calibrado
   │                             a confidence_level = critical_ratio por SKU
   ▼
src/optimization/newsvendor.py ── Q* = upper bound conformal
                                  cantidad_pedido = max(0, ⌈Q*⌉ − stock_actual)
```

**Validación.** `src/models/validator.py` implementa cross-validation temporal con 5 folds expanding-window. Cada fold se divide cronológicamente en `pure_train | early_stopping | calibration | val` para que el conjunto usado en early stopping sea disjunto del usado para calibrar el intervalo conformal.

**Salidas en `outputs/`:**

- `predictions_week_next.csv` — pronóstico diario para la semana W+1 (1 120 filas).
- `purchase_orders.csv` — cantidad de pedido por SKU-tienda con trazabilidad completa: forecast, Q* objetivo, stock actual, critical ratio, cantidad final, alertas (160 filas).

## Estructura del repositorio

```
.
├── README.md
├── Makefile
├── pytest.ini
├── requirements.txt
├── .gitignore
│
├── data/                            # Datos de entrada (5 CSVs)
│   ├── ventas_historicas.csv
│   ├── inventario_actual.csv
│   ├── catalogo_productos.csv
│   ├── maestro_tiendas.csv
│   └── ground_truth_trends.csv
│
├── src/
│   ├── config.py                    # constantes, schemas, FEATURE_COLS
│   ├── main.py                      # pipeline end-to-end + CLI
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── forecaster.py
│   │   ├── conformal.py
│   │   └── validator.py
│   └── optimization/
│       └── newsvendor.py
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_optimization_results.ipynb
│
├── outputs/
│   ├── predictions_week_next.csv
│   └── purchase_orders.csv
│
└── tests/
    ├── test_preprocessor.py
    └── test_newsvendor.py
```

## Stack

| Capa | Librería | Versión |
|---|---|---|
| Manipulación de datos | pandas | 3.0.2 |
| Cálculo numérico | numpy | 2.4.4 |
| Modelo ML | lightgbm | 4.6.0 |
| Conformal Prediction | mapie | 1.4.0 |
| Benchmark | prophet | 1.3.0 |
| Estadística clásica | statsmodels | 0.14.6 |
| Calendario | holidays | 0.95 |
| Visualización | matplotlib + seaborn | 3.10.9 / 0.13.2 |
| Tests | pytest | 9.0.3 |
| Lint | ruff | 0.15.12 |

Versiones fijadas en `requirements.txt`.

## Autor

Ana María Montes Franco
