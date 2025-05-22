#!/bin/bash
# FinFlow startup script

# Set default environment
ENV=${1:-"development"}
MODE=${2:-"server"}
PORT=${3:-"8000"}

echo "Starting FinFlow in $ENV mode as $MODE on port $PORT"

# Create log directory if it doesn't exist
mkdir -p logs

# Set environment variable
export FINFLOW_ENV=$ENV

# Start application
python main.py --env $ENV --mode $MODE --port $PORT --host 0.0.0.0
