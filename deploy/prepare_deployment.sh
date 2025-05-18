#!/bin/bash

# Create deployment package directory
mkdir -p deployment_package
mkdir -p deployment_package/logs

# Copy all necessary files
cp -r models/ deployment_package/
cp -r .streamlit/ deployment_package/
cp api.py app.py application.py app_insights.py init_models.py requirements.txt deployment_package/
cp air-paradis-logo.png deployment_package/

# Create updated startup script
cat > deployment_package/startup.sh << 'EOL'
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

# Start the Streamlit app
echo "Starting Streamlit app..."
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
EOL

chmod +x deployment_package/startup.sh

# Create web.config file for Azure
cat > deployment_package/web.config << 'EOL'
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <system.webServer>
    <handlers>
      <add name="httpPlatformHandler" path="*" verb="*" modules="httpPlatformHandler" resourceType="Unspecified" />
    </handlers>
    <httpPlatform processPath="%home%\site\wwwroot\startup.sh"
                  arguments=""
                  stdoutLogEnabled="true"
                  stdoutLogFile="%home%\LogFiles\stdout"
                  startupTimeLimit="180">
      <environmentVariables>
        <environmentVariable name="PORT" value="%HTTP_PLATFORM_PORT%" />
        <environmentVariable name="PYTHONPATH" value="%home%\site\wwwroot" />
      </environmentVariables>
    </httpPlatform>
  </system.webServer>
</configuration>
EOL

# Create a modified .deployment file
cat > deployment_package/.deployment << 'EOL'
[config]
command = bash startup.sh
EOL

# Create a runtime.txt file to specify Python version
echo "python-3.9" > deployment_package/runtime.txt

echo "Deployment package prepared successfully!"
