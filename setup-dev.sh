#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Setting up development environment..."
python -m venv venv
source venv/bin/activate

echo "Installing development dependencies..."
pip install -r requirements-dev.txt

echo "Installing pre-commit..."
pip install pre-commit

echo "Setting up pre-commit hooks..."
pre-commit install

echo "Development environment setup complete."


#chmod +x setup-dev.sh
#./setup-dev.sh