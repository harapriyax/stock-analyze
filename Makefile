# ==============================================================================
# AlphaPulse v2.0 - Make Automation
# ==============================================================================

PYTHON = .venv/Scripts/python.exe
STREAMLIT = .venv/Scripts/streamlit.exe
PYTEST = .venv/Scripts/pytest.exe
RUFF = .venv/Scripts/ruff.exe

.PHONY: help install download train train-all benchmark app test lint

help:
	@echo "AlphaPulse v2.0 Automation Targets:"
	@echo "  make install    - Install all dependencies from requirements.txt"
	@echo "  make download   - Download 25-year history for all 12 target stocks"
	@echo "  make train      - Train default model (LSTM on RELIANCE.NS)"
	@echo "  make train-all  - Run full benchmark (all 4 models on all 12 stocks)"
	@echo "  make benchmark  - Run model comparison on RELIANCE.NS"
	@echo "  make app        - Launch Streamlit interactive intelligence dashboard"
	@echo "  make test       - Execute pytest test suite with coverage"
	@echo "  make lint       - Run ruff code quality checks"

install:
	$(PYTHON) -m pip install -r requirements.txt

download:
	$(PYTHON) scripts/download_all_stocks.py

train:
	$(PYTHON) train.py --ticker RELIANCE.NS --model lstm --epochs 35

train-all:
	$(PYTHON) train.py --ticker all --model all --epochs 35

benchmark:
	$(PYTHON) train.py --ticker RELIANCE.NS --model all --epochs 35

app:
	$(STREAMLIT) run app.py

test:
	$(PYTEST) tests/ -v

lint:
	$(RUFF) check src/ tests/

