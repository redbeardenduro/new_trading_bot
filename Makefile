# Trading Bot Development Makefile
# =================================

.PHONY: help install install-dev test lint format type-check security clean setup-pre-commit

# Default target
help:
	@echo "Available targets:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  test            Run tests with coverage"
	@echo "  lint            Run linting checks"
	@echo "  format          Format code with black and isort"
	@echo "  type-check      Run type checking with mypy"
	@echo "  type-check-strict Run strict type checking"
	@echo "  security        Run security checks"
	@echo "  setup-pre-commit Setup pre-commit hooks"
	@echo "  clean           Clean up generated files"
	@echo "  ci              Run all CI checks locally"

# Installation
install:
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt

# Testing
test:
	pytest tests/ --cov=core --cov=integrations --cov=utils --cov=common \
		--cov-report=xml --cov-report=html --cov-report=term-missing

# Code Quality
lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	black .
	isort .

format-check:
	black --check --diff .
	isort --check-only --diff .

# Type Checking
type-check:
	mypy core/ utils/ common/ integrations/ai/ integrations/data/ --config-file mypy.ini

type-check-strict:
	@echo "Running strict type checking (may fail until all issues are resolved)..."
	@cp mypy.ini mypy-strict.ini
	@sed -i 's/# strict = True/strict = True/' mypy-strict.ini
	@mypy core/ utils/ common/ integrations/ai/ integrations/data/ --config-file mypy-strict.ini || true
	@rm -f mypy-strict.ini

# Security
security:
	bandit -r . --severity-level medium
	@echo "Consider running: pip-audit" 

# Pre-commit
setup-pre-commit:
	pre-commit install
	pre-commit install --hook-type commit-msg

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	rm -f bandit-report.json coverage.xml

# CI simulation
ci: format-check lint type-check security test
	@echo "All CI checks passed!"

# Development setup
setup: install-dev setup-pre-commit
	@echo "Development environment setup complete!"
	@echo "Run 'make help' to see available commands."
