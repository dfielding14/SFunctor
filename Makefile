.PHONY: help format lint type-check check clean install install-dev test

help:  ## Show this help
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	pip install -e .

install-dev:  ## Install the package with development dependencies
	pip install -e ".[dev]"

format:  ## Format code using black and isort
	@echo "Sorting imports..."
	isort sfunctor/ *.py
	@echo "Formatting code..."
	black sfunctor/ *.py
	@echo "Formatting complete!"

lint:  ## Run flake8 linting
	@echo "Running flake8..."
	flake8 sfunctor/ *.py

type-check:  ## Run mypy type checking
	@echo "Running mypy..."
	mypy sfunctor/ *.py --ignore-missing-imports

check: lint type-check  ## Run all checks (lint + type-check)
	@echo "All checks passed!"

format-check:  ## Check if code is formatted correctly without changing files
	@echo "Checking import order..."
	isort --check-only --diff sfunctor/ *.py
	@echo "Checking code format..."
	black --check --diff sfunctor/ *.py

clean:  ## Clean up generated files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/

test:  ## Run tests (placeholder for now)
	@echo "No tests configured yet. Add pytest when tests are written."

all: format check  ## Format code and run all checks 