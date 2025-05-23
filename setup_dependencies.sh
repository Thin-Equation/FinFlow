#!/bin/bash
# Script to set up and validate dependencies for the FinFlow project

set -e  # Exit immediately if any command fails

# Color codes for better readability
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}FinFlow Dependency Management Assistant${NC}"
echo -e "${GREEN}========================================${NC}"

# Check for Python 3.10+
echo -e "\n${YELLOW}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version | cut -d " " -f 2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d "." -f 1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d "." -f 2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo -e "${RED}Error: Python 3.10+ is required. Found Python $PYTHON_VERSION${NC}"
    exit 1
else
    echo -e "${GREEN}Using Python $PYTHON_VERSION${NC}"
fi

# Check if virtual environment is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "\n${YELLOW}No active virtual environment detected.${NC}"
    
    # Check if finflow-env exists
    if [ -d "finflow-env" ]; then
        echo -e "${YELLOW}Found existing finflow-env. Activating...${NC}"
        source finflow-env/bin/activate
    else
        echo -e "${YELLOW}Creating new virtual environment (finflow-env)...${NC}"
        python3 -m venv finflow-env
        source finflow-env/bin/activate
    fi
    
    echo -e "${GREEN}Virtual environment activated: $VIRTUAL_ENV${NC}"
else
    echo -e "\n${GREEN}Using existing virtual environment: $VIRTUAL_ENV${NC}"
fi

# Install/update pip
echo -e "\n${YELLOW}Updating pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies from requirements.txt...${NC}"
pip install -r requirements.txt

# Run dependency check
echo -e "\n${YELLOW}Verifying dependencies...${NC}"
chmod +x ./scripts/check_dependencies.py
./scripts/check_dependencies.py

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}All dependencies are properly installed and configured!${NC}"
    
    # Generate a frozen requirements file for reproducibility
    echo -e "\n${YELLOW}Generating frozen requirements file...${NC}"
    pip freeze > requirements.frozen.txt
    echo -e "${GREEN}Created requirements.frozen.txt for reproducible builds${NC}"
    
    echo -e "\n${GREEN}Setup complete! Your environment is ready for development.${NC}"
    
    # Instructions for new users
    echo -e "\n${YELLOW}To activate this environment in the future, run:${NC}"
    echo -e "source finflow-env/bin/activate"
else
    echo -e "\n${RED}Some dependencies may be missing. Please review the output above.${NC}"
    exit 1
fi
