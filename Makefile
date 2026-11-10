PY ?= .venv/bin/python
BCP ?= .venv/bin/bcp

.PHONY: install data all sample test lint fixture api mlflow

install:
	python3.13 -m venv .venv && $(PY) -m pip install -e ".[api,dev]"

# raw data: 8.3 GB MATR + 0.2 GB NASA, checksums in data/manifest.json
data:
	$(PY) scripts/download_data.py
	mkdir -p data/raw/nasa/extracted && cd data/raw/nasa/extracted && unzip -q -o "../5. Battery Data Set.zip" \
	  && cd "5. Battery Data Set" && for z in *.zip; do mkdir -p "$${z%.zip}" && unzip -q -o "$$z" -d "$${z%.zip}"; done

# every number in reports/results.md (and the README) comes from this
all:
	$(BCP) all --raw data/raw/matr --nasa data/raw/nasa/extracted --root data/processed --reports reports --workers 4

sample:
	$(BCP) all --raw tests/fixtures --nasa tests/fixtures/nasa --root out/sample --workers 2

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check src tests scripts

fixture:
	$(PY) scripts/make_fixture.py --raw data/raw/matr --out tests/fixtures

api:
	BCP_ROOT=data/processed .venv/bin/uvicorn battery_pipeline.api:app --port 8000

mlflow:
	.venv/bin/mlflow ui --backend-store-uri sqlite:///data/processed/mlflow.db --port 5000
