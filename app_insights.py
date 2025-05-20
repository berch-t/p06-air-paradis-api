import os
import requests
import datetime
import json
import logging

# Configuration pour Application Insights
APPINSIGHTS_INSTRUMENTATION_KEY = os.environ.get('APPINSIGHTS_INSTRUMENTATION_KEY', '9af56b4d-4ad5-4643-ba29-41d154893ad4')
APPINSIGHTS_ENDPOINT = "https://dc.services.visualstudio.com/v2/track"

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_insights.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def send_event(event_name, properties=None):
    """
    Envoie un événement personnalisé à Application Insights
    """
    if not APPINSIGHTS_INSTRUMENTATION_KEY:
        logger.warning("Clé d'instrumentation Application Insights non configurée.")
        return False
    
    try:
        # Création du payload pour Application Insights
        payload = {
            "name": "Microsoft.ApplicationInsights.Event",
            "time": datetime.datetime.utcnow().isoformat() + "Z",
            "iKey": APPINSIGHTS_INSTRUMENTATION_KEY,
            "tags": {
                "ai.cloud.roleInstance": "sentiment-analysis-api"
            },
            "data": {
                "baseType": "EventData",
                "baseData": {
                    "ver": 2,
                    "name": event_name,
                    "properties": properties or {}
                }
            }
        }
        
        # Envoi de la télémétrie
        response = requests.post(APPINSIGHTS_ENDPOINT, json=payload)
        if response.status_code != 200:
            logger.warning(f"Erreur lors de l'envoi à Application Insights: {response.status_code} - {response.text}")
            return False
        else:
            logger.info(f"Événement '{event_name}' envoyé à Application Insights")
            return True
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi à Application Insights: {str(e)}")
        return False

def send_to_appinsights(tweet, prediction, is_incorrect=False):
    """
    Envoie des données de télémétrie à Azure Application Insights
    """
    event_name = "IncorrectPrediction" if is_incorrect else "Prediction"
    properties = {
        "tweet": tweet,
        "predicted_sentiment": prediction,
        "is_incorrect": str(is_incorrect)
    }
    
    return send_event(event_name, properties)

# Test de l'envoi à Application Insights au démarrage
if __name__ == "__main__":
    success = send_event("AppStarted", {"status": "Application démarrée avec succès"})
    print(f"Test d'envoi à Application Insights: {'Réussi' if success else 'Échoué'}")