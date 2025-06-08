#!/bin/bash

# Medical Image Classification - Frontend Startup Script

echo "Starting Medical Image Classification Frontend..."

# Navigate to project directory
cd "$(dirname "$0")"

# Navigate to frontend directory
cd FrontEnd

# Check if index1.html exists
if [ ! -f "index1.html" ]; then
    echo "ERROR: index1.html not found in FrontEnd directory!"
    exit 1
fi

# Start Python HTTP server for frontend
echo "Starting Frontend server on http://127.0.0.1:3000"
echo "Open http://127.0.0.1:3000/index1.html in your browser"
echo ""
echo "Make sure the backend is running on http://127.0.0.1:8000"
echo "Press Ctrl+C to stop the server"
python3 -m http.server 3000 