.PHONY: install eda train predict test lint all clean help

# Resolve Python from .venv si existe, sino usar el del sistema.
PYTHON := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; \
                   elif [ -x .venv/Scripts/python.exe ]; then echo .venv/Scripts/python.exe; \
                   else echo python; fi)

help:
	@echo "Targets disponibles:"
	@echo "  install   - Instala dependencias en .venv (uv venv + uv pip)"
	@echo "  eda       - Ejecuta notebooks 01 y 02 in-place"
	@echo "  train     - Ejecuta CV completo via notebook 03"
	@echo "  predict   - Corre el pipeline end-to-end (genera CSVs en outputs/)"
	@echo "  test      - pytest -v"
	@echo "  lint      - ruff check + format"
	@echo "  all       - clean + install + test + predict + eda + train"
	@echo "  clean     - Borra outputs y caches"

install:
	uv venv --python 3.11 .venv
	uv pip install --python $(PYTHON) -r requirements.txt

eda:
	$(PYTHON) -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/01_EDA.ipynb
	$(PYTHON) -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/02_feature_engineering.ipynb

train:
	$(PYTHON) -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 notebooks/03_modeling.ipynb

predict:
	$(PYTHON) -m src.main

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m ruff format src/ tests/ --check

all: clean test predict eda train
	$(PYTHON) -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/04_optimization_results.ipynb
	@echo ""
	@echo "Pipeline completo OK. Ver outputs/ y notebooks/ ejecutados."

clean:
	rm -rf outputs/*.csv outputs/*.png .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -prune -exec rm -rf {} +
