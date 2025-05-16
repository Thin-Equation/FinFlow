#!/bin/bash
# Run the agent communication test

# Activate virtual environment if it exists
if [ -d "finflow-env" ]; then
    echo "Activating virtual environment..."
    source finflow-env/bin/activate
fi

# Run the test
echo "Running agent communication test..."
python -m tests.test_agent_communication

# Print completion message
echo "Agent communication test completed."
