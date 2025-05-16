#!/usr/bin/env bash

# Script to run the Hello World agent using ADK CLI

# Activate virtual environment if not already active
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "Activating virtual environment..."
    source ./finflow-env/bin/activate
fi

# Check if ADK CLI is installed
if ! command -v adk >/dev/null 2>&1; then
    echo "Error: ADK CLI not found. Please install it using:"
    echo "pip install google-adk"
    exit 1
fi

# Default values
AGENT_PATH="/Users/dhairyagundechia/Downloads/finflow/agents/hello_world_agent.py"
PORT=8080
DEBUG=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port|-p)
            PORT="$2"
            shift
            ;;
        --debug|-d)
            DEBUG=true
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
    shift
done

# Print startup message
echo "Starting Hello World agent on port $PORT..."
echo "Agent path: $AGENT_PATH"

# Run the agent with ADK CLI
if [ "$DEBUG" = true ]; then
    echo "Running in debug mode..."
    adk run --model-type=py --path="$AGENT_PATH" --port="$PORT" --verbose
else
    adk run --model-type=py --path="$AGENT_PATH" --port="$PORT"
fi
