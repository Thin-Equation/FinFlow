#!/bin/bash
# Script to run the agent communication example

# Activate virtual environment if it exists
if [ -d "finflow-env" ]; then
    echo "Activating virtual environment..."
    source finflow-env/bin/activate
fi

# Create examples directory if it doesn't exist
if [ ! -d "examples" ]; then
    mkdir -p examples
fi

# Run the example
echo "Running agent communication example..."
python -m examples.agent_communication_example

# Print completion message
echo "Agent communication example completed."
