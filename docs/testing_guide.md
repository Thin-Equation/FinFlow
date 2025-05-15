# FinFlow Agent Testing Guide

This guide provides instructions for testing the FinFlow agents using both unit tests and the ADK CLI.

## Prerequisites

- Python 3.9+ (3.13.3 recommended)
- Virtual environment (finflow-env)
- Google ADK installed (v0.5.0)

## Setting Up the Test Environment

1. **Activate the virtual environment**:
   ```bash
   source ./finflow-env/bin/activate
   ```

2. **Install test dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov
   ```

## Running Unit Tests

### Running the Hello World Test

The Hello World test verifies basic agent functionality:

```bash
# Run with verbose output
pytest tests/test_hello_world_agent.py -v

# Run with coverage
pytest tests/test_hello_world_agent.py --cov=agents.base_agent
```

### Running Agent Initialization Tests

These tests verify agent initialization and configuration:

```bash
pytest tests/test_agent_initialization.py -v
```

### Running All Tests

To run all unit tests:

```bash
pytest
```

## Testing with ADK CLI

The ADK CLI allows interactive testing of agents. We provide a script for running the Hello World agent:

### Hello World Agent Testing

1. **Run the Hello World agent**:
   ```bash
   # Make sure the script is executable
   chmod +x ./run_hello_world.sh
   
   # Run the agent on the default port (8080)
   ./run_hello_world.sh
   ```

2. **Using a different port**:
   ```bash
   ./run_hello_world.sh --port 8081
   ```

3. **Running in debug mode**:
   ```bash
   ./run_hello_world.sh --debug
   ```

4. **Interacting with the agent**:
   Once the agent is running, you can interact with it:
   - In a web browser, go to http://localhost:8080
   - Use the ADK CLI interface
   - Send requests via curl:
     ```bash
     curl -X POST http://localhost:8080 \
       -H "Content-Type: application/json" \
       -d '{"input": "Hello"}'
     ```

## Test Coverage and Reporting

To generate a coverage report:

```bash
pytest --cov=agents --cov-report=html
```

This will generate an HTML report in the `htmlcov/` directory.

## Troubleshooting

- **ImportError with Google ADK**: Ensure the ADK is installed correctly with `pip install google-adk==0.5.0`
- **Module not found errors**: Check that you're running tests from the project root directory
- **Permission denied for run_hello_world.sh**: Run `chmod +x run_hello_world.sh`

## Next Steps

After confirming basic agent functionality:

1. Implement and test specialized agent capabilities
2. Add integration tests for agent communication
3. Test with real document samples
4. Implement more complex agent workflows
