#!/bin/bash
# Script to run the agent communication mock example

# Activate virtual environment if it exists
if [ -d "finflow-env" ]; then
    echo "Activating virtual environment..."
    source finflow-env/bin/activate
fi

# Run the example
echo "Running agent communication mock example..."
./finflow-env/bin/python3 examples/agent_communication_mock_example.py

# Print completion message
echo "Agent communication mock example completed."
