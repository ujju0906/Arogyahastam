#!/bin/bash

# Medical Image Classification - Backend Startup Script

echo "Starting Medical Image Classification Backend..."

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
echo "Activating virtual environment..."
source env/bin/activate

# Navigate to backend directory
cd Backend

# Check if model file exists
if [ ! -f "../Models/densenet_final_model.pt" ] && [ ! -f "../Models/augmented_densenet_final_model.pt" ]; then
    echo "ERROR: Model file not found in Models directory!"
    echo "Please ensure either 'densenet_final_model.pt' or 'augmented_densenet_final_model.pt' exists in the Models folder."
    exit 1
fi

# Start the FastAPI server
echo "Starting FastAPI server on http://127.0.0.1:8000"
echo "API Documentation will be available at http://127.0.0.1:8000/docs"
uvicorn app:app --host 127.0.0.1 --port 8000 --reload 