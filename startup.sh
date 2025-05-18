#!/bin/bash

# Set environment variables
export APPINSIGHTS_INSTRUMENTATION_KEY=9af56b4d-4ad5-4643-ba29-XXXXXXXXXXXX
export API_URL=http://localhost:5000
export API_URL_PROD=air-paradis-sentiment-api-dkceasgya2cvaehc.francecentral-01.azurewebsites.net
export MODEL_PATH=models
export PORT=8000
export PYTHONPATH=/home/site/wwwroot

# Display directory contents for debugging
echo "Current directory: $(pwd)"
echo "Directory contents:"
ls -la

# Create log directory if it doesn't exist
mkdir -p logs

# Ensure dependencies are installed
echo "Installing required packages..."
pip install -r requirements.txt
pip install gunicorn streamlit

# Start the API directly with Python instead of gunicorn
echo "Starting API server..."
python application.py > logs/api.log 2>&1 &
API_PID=$!
echo "API server started with PID: $API_PID"

# Wait for API to initialize
echo "Waiting for API to initialize..."
sleep 20

# Check if API process is still running
if ps -p $API_PID > /dev/null; then
    echo "API server is running correctly"
else
    echo "Warning: API server process exited. Checking logs:"
    cat logs/api.log
    # Try to start again with more debug info
    echo "Trying to start API again with more debug..."
    FLASK_DEBUG=1 FLASK_ENV=development python application.py > logs/api_debug.log 2>&1 &
    API_PID=$!
    sleep 10
fi

# Verify streamlit is available
which streamlit || echo "Streamlit not found in PATH"
pip list | grep streamlit

# Start the Streamlit app
echo "Starting Streamlit app..."
python -m streamlit run app.py --server.port=8501 --server.address=0.0.0.0