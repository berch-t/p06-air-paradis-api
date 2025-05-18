"""
Configuration pour Azure Application Insights
Ce module gère la configuration et l'envoi de traces vers Azure Application Insights
"""

import os
import json
import logging
import requests
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("appinsights.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Récupération de la clé d'instrumentation depuis les variables d'environnement
# ou depuis un fichier .env avec python-dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv non installé, utilisation des variables d'environnement uniquement")

APPINSIGHTS_INSTRUMENTATION_KEY = os.getenv('APPINSIGHTS_INSTRUMENTATION_KEY', '')
APPINSIGHTS_ENDPOINT = "https://dc.services.visualstudio.com/v2/track"

def configure_app_insights():
    """
    Configure Azure Application Insights et retourne la clé d'instrumentation
    """
    if not APPINSIGHTS_INSTRUMENTATION_KEY:
        logger.warning("Clé d'instrumentation Azure Application Insights non définie.")
        logger.warning("Définissez la variable d'environnement APPINSIGHTS_INSTRUMENTATION_KEY ou ajoutez-la dans un fichier .env")
    
    return APPINSIGHTS_INSTRUMENTATION_KEY

def send_to_appinsights(tweet, prediction, is_incorrect=False, properties=None):
    """
    Envoie des données de télémétrie à Azure Application Insights
    
    Arguments:
        tweet (str): Le tweet analysé
        prediction (str): La prédiction de sentiment
        is_incorrect (bool): Indique si la prédiction est incorrecte selon le feedback utilisateur
        properties (dict): Propriétés supplémentaires à inclure dans la trace
    """
    if not APPINSIGHTS_INSTRUMENTATION_KEY:
        logger.warning("Clé d'instrumentation Azure Application Insights non configurée.")
        return False
    
    try:
        # Initialisation des propriétés
        custom_properties = {
            "tweet": tweet,
            "predicted_sentiment": prediction,
            "is_incorrect": str(is_incorrect)
        }
        
        # Ajout des propriétés supplémentaires
        if properties:
            custom_properties.update(properties)
        
        # Création du payload pour Application Insights
        payload = {
            "name": "Microsoft.ApplicationInsights.Event",
            "time": datetime.utcnow().isoformat() + "Z",
            "iKey": APPINSIGHTS_INSTRUMENTATION_KEY,
            "tags": {
                "ai.cloud.roleInstance": "sentiment-analysis-api",
                "ai.device.os": "Web"
            },
            "data": {
                "baseType": "EventData",
                "baseData": {
                    "ver": 2,
                    "name": "IncorrectPrediction" if is_incorrect else "Prediction",
                    "properties": custom_properties,
                    "measurements": {
                        "processingTime": 0.0  # Vous pouvez ajouter le temps de traitement réel ici
                    }
                }
            }
        }
        
        # Envoi de la télémétrie
        response = requests.post(APPINSIGHTS_ENDPOINT, json=payload)
        
        if response.status_code != 200:
            logger.warning(f"Erreur lors de l'envoi à Application Insights: {response.status_code} - {response.text}")
            return False
            
        logger.info(f"Trace envoyée à Application Insights: {'IncorrectPrediction' if is_incorrect else 'Prediction'}")
        return True
            
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi à Application Insights: {str(e)}")
        return False