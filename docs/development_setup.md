# Development Setup Guide

This document provides comprehensive instructions for setting up a development environment for the FinFlow project.

## Initial Setup

### Prerequisites

Before starting development on FinFlow, ensure you have the following prerequisites:

- Python 3.10 or higher
- pip (Python package manager)
- Git
- Google Cloud SDK
- A Google Cloud project with:
  - Vertex AI API enabled
  - Document AI API enabled
  - BigQuery API enabled
  - Cloud Storage API enabled

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/finflow.git
   cd finflow
   ```

2. Set up your development environment:
   ```bash
   # Option 1: Use the automated setup script (recommended)
   ./setup_dependencies.sh
   
   # Option 2: Manual setup
   python -m venv finflow-env
   source finflow-env/bin/activate  # On Windows: finflow-env\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configure Google Cloud authentication:
   ```bash
   gcloud auth application-default login
   ```

## Dependency Management

### Understanding the Dependency Structure

FinFlow uses several dependency files:

- **requirements.txt**: Main dependencies for development and production
- **requirements.frozen.txt**: Exact versions for reproducible builds
- **scripts/check_dependencies.py**: Tool to verify dependencies

### Adding New Dependencies

When adding new functionality that requires additional packages:

1. Add the dependency to requirements.txt with an appropriate version constraint
   ```
   new-package>=1.0.0
   ```

2. Update your local environment:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify dependencies are properly tracked:
   ```bash
   ./scripts/check_dependencies.py
   ```

4. Update the frozen requirements (optional but recommended):
   ```bash
   pip freeze > requirements.frozen.txt
   ```

### Updating Dependencies

To update all dependencies to their latest compatible versions:

```bash
# Use the Makefile target
make update-deps

# Or manually
pip install --upgrade -r requirements.txt
pip freeze > requirements.frozen.txt
```

### Troubleshooting Dependency Issues

If you encounter package conflicts or import errors:

1. Verify your virtual environment is active
2. Run the dependency checker:
   ```bash
   ./scripts/check_dependencies.py
   ```
3. Check for conflicting versions:
   ```bash
   pip check
   ```
4. If necessary, recreate your virtual environment:
   ```bash
   deactivate
   rm -rf finflow-env
   ./setup_dependencies.sh
   ```

## Development Workflow

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and run tests:
   ```bash
   make test
   ```

3. Run code quality checks:
   ```bash
   make lint
   ```

4. Commit your changes:
   ```bash
   git commit -am "Add your descriptive commit message"
   ```

5. Push to your branch:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Create a pull request for review

## Common Tasks

### Running the Application

```bash
# Development mode with CLI
./start.sh development cli

# Development mode with server
./start.sh development server 8000

# Production mode
./start.sh production server 8000
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test suite
pytest tests/test_document_processor_agent.py

# Run end-to-end tests
make test-e2e
```

### Building Docker Container

```bash
# Build the Docker image
make docker

# Run with Docker Compose
make docker-compose
```
