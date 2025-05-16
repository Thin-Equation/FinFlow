#!/usr/bin/env bash

# Script to run ADK CLI for local testing of FinFlow agents

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
AGENT_NAME="FinFlow_MasterOrchestrator"
AGENT_PATH="./agents/master_orchestrator.py"
PORT=8080
DEBUG=false
CLEAR_CACHE=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --agent|-a)
            AGENT_NAME="$2"
            shift
            if [[ "$AGENT_NAME" == "FinFlow_MasterOrchestrator" ]]; then
                AGENT_PATH="./agents/master_orchestrator.py"
            elif [[ "$AGENT_NAME" == "FinFlow_DocumentProcessor" ]]; then
                AGENT_PATH="./agents/document_processor.py"
            elif [[ "$AGENT_NAME" == "FinFlow_Validation" ]]; then
                AGENT_PATH="./agents/validation_agent.py"
            elif [[ "$AGENT_NAME" == "FinFlow_RuleRetrieval" ]]; then
                AGENT_PATH="./agents/rule_retrieval.py"
            elif [[ "$AGENT_NAME" == "FinFlow_Storage" ]]; then
                AGENT_PATH="./agents/storage_agent.py"
            elif [[ "$AGENT_NAME" == "FinFlow_Analytics" ]]; then
                AGENT_PATH="./agents/analytics_agent.py"
            else
                echo "Error: Unknown agent $AGENT_NAME"
                exit 1
            fi
            ;;
        --port|-p)
            PORT="$2"
            shift
            ;;
        --debug|-d)
            DEBUG=true
            ;;
        --clear-cache|-c)
            CLEAR_CACHE=true
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --agent, -a     Agent name to run (default: FinFlow_MasterOrchestrator)"
            echo "  --port, -p      Port to run the server on (default: 8080)"
            echo "  --debug, -d     Enable debug mode"
            echo "  --clear-cache, -c  Clear ADK cache before running"
            echo "  --help, -h      Display this help message"
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
    shift
done

# Export environment variables
export FINFLOW_ENV=development
export ADK_DEBUG=$DEBUG

# Clear cache if requested
if [[ "$CLEAR_CACHE" = true ]]; then
    echo "Clearing ADK cache..."
    adk clear-cache
fi

# Run ADK CLI
echo "Starting $AGENT_NAME agent on port $PORT..."
adk run agent --agent $AGENT_PATH --port $PORT
