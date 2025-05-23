# FinFlow - Financial Document Processing Platform

A production-ready financial document processing and analysis platform built with Google's Agent Development Kit (ADK) and Google Cloud services.

## Overview
FinFlow is an intelligent system that analyzes, processes, and extracts insights from financial documents using multiple specialized agents. The system leverages Google Document AI for document processing, BigQuery for data storage, and Vertex AI Gemini models for intelligent processing.

This project provides a complete end-to-end integration of all agent components into a production-level implementation with multiple deployment options, comprehensive testing, and robust configuration management.

## Features
- Multi-agent architecture for specialized document processing tasks
- Advanced agent communication framework with state management
- Intelligent task delegation across specialized agents
- Document parsing and entity extraction using Document AI
- Validation against business rules and compliance requirements
- Structured data storage in BigQuery
- Financial analytics and insights generation
- Comprehensive agent testing framework

## Agent Architecture
The system consists of several specialized agents:

1. **Master Orchestrator** - Coordinates the overall workflow
2. **Document Processor** - Extracts information from documents
3. **Validation** - Validates documents against rules
4. **Storage** - Manages data persistence in BigQuery and other storage systems
5. **Analytics** - Generates financial insights

For more details, see [Agent Architecture](docs/agent_architecture.md).

## Agent Communication Framework

The FinFlow system features a robust Agent Communication Framework that provides:

- **Full Communication Protocol** - Standardized message passing with delivery guarantees
- **Task Execution Framework** - Task creation, tracking, and hierarchical execution
- **State-based Communication** - Workflow state management with tracking and history
- **LLM-driven Delegation** - Intelligent task delegation based on agent capabilities
- **Advanced Delegation Strategies** - Multiple strategies for optimal agent selection

For more details, see [Agent Communication Framework](docs/agent_communication_framework.md).

## Development

For comprehensive setup instructions and development guidelines, see:
- [Development Setup Guide](docs/development_setup.md)
- [Agent Architecture](docs/agent_architecture.md)
- [Optimization Guide](docs/optimization_guide.md)

### Storage Agent
The StorageAgent is responsible for all database operations in the FinFlow system. It provides:

- Comprehensive BigQuery dataset and schema management
- Document and entity storage with optimized data access patterns
- Financial data analysis and reporting capabilities
- Relationship tracking between financial documents
- Caching layer for performance optimization

For more details, see [Storage Agent Documentation](docs/storage_agent.md).

## Running the System

FinFlow can be run in several modes to accommodate different use cases:

### Server Mode
Run as a web API server:
```bash
./start.sh production server 8000
```

Or using Python directly:
```bash
python main.py --env production --mode server --port 8000
```

### CLI Mode
Run in interactive command-line mode:
```bash
./start.sh production cli
```

Or using Python directly:
```bash
python main.py --env production --mode cli
```

### Batch Processing Mode
Process a directory of documents:
```bash
python main.py --env production --mode batch --batch-dir /path/to/documents
```

### Workflow Mode
Run a specific workflow against a document:
```bash
python main.py --env production --mode workflow --workflow invoice --document /path/to/invoice.pdf
```

## Docker Deployment
The system can be deployed using Docker:

```bash
# Build the Docker image
docker build -t finflow .

# Run with Docker Compose for full environment
docker-compose up -d
```

## Setup

### Prerequisites
- Python 3.10+
- Google Cloud project with the following APIs enabled:
  - Vertex AI API
  - Document AI API
  - BigQuery API
  - Cloud Storage API

### Installation
1. Clone this repository
```bash
git clone https://github.com/Thin-Equation/finflow.git
cd finflow
```

2. Create and activate a virtual environment
```bash
python -m venv finflow-env
source finflow-env/bin/activate  # On Windows: finflow-env\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up your configuration
```bash
export FINFLOW_ENV=development
# Optionally create a .env file with your configuration settings
```

### Dependency Management
The project includes several tools for dependency management:

#### Quick Setup

For new developers, the easiest way to set up the environment is with our setup script:

```bash
# Setup and validate all dependencies at once
./setup_dependencies.sh
```

This script will:
- Check your Python version
- Create or activate a virtual environment
- Install all dependencies
- Verify dependencies match project requirements
- Create a frozen requirements file for reproducible builds

#### Advanced Options

For more granular dependency management, use the Makefile targets:

```bash
# Install dependencies
make install

# Check if all dependencies are properly listed in requirements.txt
make check-deps

# Update dependencies to latest compatible versions
make update-deps
```

#### Adding New Dependencies

When adding new dependencies to the project:
1. Add them to requirements.txt with appropriate version constraints
2. Run `make check-deps` to verify all imports are properly documented
3. Consider updating the frozen requirements for reproducibility

#### Checking For Updates

To check for outdated dependencies and get recommended updates:

```bash
# Show outdated dependencies and safety analysis
make outdated-deps
# or directly
./scripts/check_outdated_deps.py
```

This tool analyzes your dependencies against semantic versioning constraints and suggests safe updates.

## Running the System
Use the provided script to run the ADK CLI:

```bash
./run_adk.sh
```

Or to run a specific agent:

```bash
./run_adk.sh --agent FinFlow_DocumentProcessor
```

## Testing
Run the unit tests:

```bash
python -m unittest discover tests
```

For more comprehensive testing options:

```bash
# Run tests with pytest
pytest

# Generate test coverage report
pytest --cov=agents --cov-report=html
```

For interactive testing with ADK CLI:
```bash
# Run the Hello World agent
./run_hello_world.sh

# Run with debug mode
./run_hello_world.sh --debug
```

For detailed testing information, see [Testing Guide](docs/testing_guide.md) and [Initial Agent Testing](docs/initial_agent_testing.md).

## Configuration
The system uses environment-specific YAML configuration files:
- `config/development.yaml` - Development settings
- `config/staging.yaml` - Staging settings
- `config/production.yaml` - Production settings

Configuration can be overridden using local files (e.g., `config/development.local.yaml`).
4. Configure authentication

## Development
- Use `pip install -r requirements.txt` to install dependencies
- Configure environment variables

## Structure
- `agents/`: Agent definitions
- `tools/`: Custom tools
- `models/`: Data models
- `config/`: Configuration files
- `tests/`: Test cases
- `utils/`: Utility functions
- `app.py`: Main application entry point
