#!/bin/bash

# Set environment variables
export APPINSIGHTS_INSTRUMENTATION_KEY=9af56b4d-4ad5-4643-ba29-XXXXXXXXXXXX
export API_URL=http://localhost:5000
export API_URL_PROD=air-paradis-sentiment-api-dkceasgya2cvaehc.francecentral-01.azurewebsites.net
export MODEL_PATH=models

# Start the API in the background
echo "Starting API server..."
nohup gunicorn --bind=0.0.0.0:5000 --timeout 600 application:app > api.log 2>&1 &

# Wait for API to initialize
echo "Waiting for API to initialize..."
sleep 10

# Start the Streamlit app
echo "Starting Streamlit app..."
streamlit run app.py --server.port=8501 --server.address=0.0.0.0