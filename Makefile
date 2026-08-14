PYTHON ?= .venv/bin/python

.PHONY: check test lint typecheck train-router demo

check: lint typecheck test

test:
	$(PYTHON) -m pytest --cov=metrics --cov=retriever --cov=routing --cov-report=term-missing --cov-fail-under=90

lint:
	$(PYTHON) -m ruff check src tests demo

typecheck:
	$(PYTHON) -m mypy -m metrics -m retriever -m routing -m train_router

train-router:
	$(PYTHON) src/train_router.py

demo:
	$(PYTHON) -m streamlit run demo/app.py
