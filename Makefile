PYTHON ?= .venv/bin/python

.PHONY: check test lint typecheck train-router demo

check: lint typecheck test

test:
	$(PYTHON) -m pytest --cov=metrics --cov=retriever --cov=routing --cov=answer_ablation --cov=outcome_routing --cov=experiment_budget --cov=reliability_study --cov=reliability_analysis --cov-report=term-missing --cov-fail-under=90

lint:
	$(PYTHON) -m ruff check src tests demo

typecheck:
	$(PYTHON) -m mypy -m metrics -m retriever -m routing -m train_router -m significance -m answer_ablation -m outcome_routing -m experiment_budget -m reliability_study -m reliability_analysis

train-router:
	$(PYTHON) src/train_router.py

demo:
	$(PYTHON) -m streamlit run demo/app.py
