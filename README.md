# FinFlow - Financial Document Processing System

A financial document processing and analysis platform built with Google's Agent Development Kit (ADK) and Google Cloud services.

## Overview
FinFlow is an intelligent system that analyzes, processes, and extracts insights from financial documents using multiple specialized agents. The system leverages Google Document AI for document processing, BigQuery for data storage, and Vertex AI Gemini models for intelligent processing.

## Features
- Multi-agent architecture for specialized document processing tasks
- Document parsing and entity extraction using Document AI
- Validation against business rules and compliance requirements
- Structured data storage in BigQuery
- Financial analytics and insights generation

## Agent Architecture
The system consists of several specialized agents:

1. **Master Orchestrator** - Coordinates the overall workflow
2. **Document Processor** - Extracts information from documents
3. **Rule Retrieval** - Retrieves compliance rules
4. **Validation** - Validates documents against rules
5. **Storage** - Manages data persistence
6. **Analytics** - Generates financial insights

For more details, see [Agent Architecture](docs/agent_architecture.md).

## Setup

### Prerequisites
- Python 3.13+
- Google Cloud project with the following APIs enabled:
  - Vertex AI API
  - Document AI API
  - BigQuery API
  - Cloud Storage API

### Installation
1. Clone this repository
```bash
git clone https://github.com/yourusername/finflow.git
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
