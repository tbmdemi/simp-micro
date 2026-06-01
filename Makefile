.PHONY: install install-core install-analysis install-dev test lint format clean run-simp run-analysis

# ─── Installation ────────────────────────────────────────────────────────────

install:
	pip install -e .

install-core:
	pip install -r requirements-core.txt

install-analysis:
	pip install -r requirements-analysis.txt

install-dev:
	pip install -r requirements-dev.txt
	pip install -e .

# ─── Testing ─────────────────────────────────────────────────────────────────

test:
	python -m pytest tests/ -v --cov=simp --cov=analysis --cov-report=term-missing

test-coverage:
	python -m pytest tests/ -v --cov=simp --cov=analysis --cov-report=html
	@echo "Coverage report: open htmlcov/index.html"

# ─── Code Quality ────────────────────────────────────────────────────────────

lint:
	flake8 simp/ analysis/ tests/
	mypy simp/ analysis/

format:
	black simp/ analysis/ tests/
	isort simp/ analysis/ tests/

check:
	black --check simp/ analysis/ tests/
	isort --check simp/ analysis/ tests/

# ─── Running ─────────────────────────────────────────────────────────────────

run-simp:
	python -m simp.main $(ARGS)

run-analysis:
	python -m analysis.cli $(ARGS)

# ─── Cleanup ─────────────────────────────────────────────────────────────────

clean:
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf *.egg-info/
	rm -rf dist/
	rm -rf build/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf simp_result_*/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ─── Help ────────────────────────────────────────────────────────────────────

help:
	@echo "Usage:"
	@echo "  make install           Install package in development mode"
	@echo "  make install-core      Install core dependencies only"
	@echo "  make install-analysis  Install analysis dependencies"
	@echo "  make install-dev       Install all dev dependencies"
	@echo "  make test              Run tests with coverage"
	@echo "  make lint              Run flake8 and mypy"
	@echo "  make format            Format code with black + isort"
	@echo "  make run-simp ARGS=... Run SIMP optimization"
	@echo "  make run-analysis      Run analysis pipeline"
	@echo "  make clean             Clean build artifacts"
