#!/bin/bash

# FinFlow Web Server Startup Script
# This script starts the FinFlow web application with the integrated UI

set -e

echo "🚀 Starting FinFlow Web Application..."

# Change to the project directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "finflow-env" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    echo "Run: python -m venv finflow-env && source finflow-env/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source finflow-env/bin/activate

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python -c "import fastapi, uvicorn" 2>/dev/null || {
    echo "❌ Required packages not found. Installing..."
    pip install -r requirements.txt
}

# Set environment variables
export FINFLOW_HOST=${FINFLOW_HOST:-127.0.0.1}
export FINFLOW_PORT=${FINFLOW_PORT:-8000}
export FINFLOW_RELOAD=${FINFLOW_RELOAD:-false}

echo "🌐 Starting server on http://${FINFLOW_HOST}:${FINFLOW_PORT}"
echo "📚 API docs will be available at http://${FINFLOW_HOST}:${FINFLOW_PORT}/docs"
echo "🎯 Web UI will be available at http://${FINFLOW_HOST}:${FINFLOW_PORT}"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the web server
python web_server.py
