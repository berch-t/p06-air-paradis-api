#!/bin/bash

# Create deployment package directory
mkdir -p deployment_package

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

# Start the API in the background
echo "Starting API server..."
cd /home/site/wwwroot/
nohup gunicorn --bind=0.0.0.0:5000 --timeout 600 application:app > api.log 2>&1 &
API_PID=$!
echo "API server started with PID: $API_PID"

# Wait for API to initialize
echo "Waiting for API to initialize..."
sleep 15

# Check if API is running
if ps -p $API_PID > /dev/null; then
    echo "API server is running correctly"
else
    echo "ERROR: API server failed to start"
    exit 1
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
                  startupTimeLimit="120">
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
echo "python-3.12" > deployment_package/runtime.txt

echo "Deployment package prepared successfully!"
