# FinFlow Makefile
# This file provides shortcuts for common development tasks

.PHONY: setup install test lint docker docker-compose server cli batch workflow clean

# Default target
all: install test

# Setup environment
setup:
	python -m venv finflow-env
	. finflow-env/bin/activate && pip install --upgrade pip

# Install dependencies
install:
	pip install -r requirements.txt

# Install server dependencies
install-server:
	pip install -r requirements-server.txt

# Run all tests
test:
	pytest

# Run specific test suite
test-e2e:
	pytest tests/test_end_to_end.py

# Run linting
lint:
	flake8 .
	mypy .
	black --check .

# Format code
format:
	black .
	isort .

# Build Docker image
docker:
	docker build -t finflow:latest .

# Run with Docker Compose
docker-compose:
	docker-compose up -d

# Run server mode
server:
	python main.py --mode server --port 8000

# Run CLI mode
cli:
	python main.py --mode cli

# Run batch mode
batch:
	python main.py --mode batch --batch-dir $(dir)

# Run workflow mode
workflow:
	python main.py --mode workflow --workflow $(workflow) --document $(doc)

# Run full end-to-end test with sample data
demo:
	python scripts/run_test_workflows.py

# Clean build artifacts and cache files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
