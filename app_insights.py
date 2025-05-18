"""
Configuration pour Azure Application Insights
Ce module gère la configuration et l'envoi de traces vers Azure Application Insights
"""

import os
import json
import logging
import requests
import time
from datetime import datetime, timedelta
from threading import Lock

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

# Configuration pour le suivi des erreurs consécutives
ERROR_THRESHOLD = 3  # Nombre d'erreurs consécutives avant alerte
ERROR_WINDOW = 300  # Fenêtre de temps en secondes (5 minutes)
NOTIFICATION_EMAIL = "berchet.thomas@gmail.com"

# Variables pour le suivi des erreurs
error_history = []
error_lock = Lock()

def configure_app_insights():
    """
    Configure Azure Application Insights et retourne la clé d'instrumentation
    """
    if not APPINSIGHTS_INSTRUMENTATION_KEY:
        logger.warning("Clé d'instrumentation Azure Application Insights non définie.")
        logger.warning("Définissez la variable d'environnement APPINSIGHTS_INSTRUMENTATION_KEY ou ajoutez-la dans un fichier .env")
    
    return APPINSIGHTS_INSTRUMENTATION_KEY

def check_consecutive_errors():
    """
    Vérifie s'il y a eu ERROR_THRESHOLD erreurs consécutives dans la fenêtre ERROR_WINDOW
    """
    with error_lock:
        if len(error_history) < ERROR_THRESHOLD:
            return False
            
        # Nettoyer les erreurs trop anciennes
        current_time = time.time()
        error_history[:] = [ts for ts in error_history if current_time - ts < ERROR_WINDOW]
        
        # Vérifier si nous avons ERROR_THRESHOLD erreurs consécutives
        if len(error_history) >= ERROR_THRESHOLD:
            # Vérifier que les erreurs sont consécutives (pas d'interruption)
            for i in range(1, len(error_history)):
                if error_history[i] - error_history[i-1] > 60:  # Plus d'une minute entre les erreurs = non consécutif
                    return False
            return True
            
        return False

def send_alert_email(properties):
    """
    Envoie une alerte par email en cas d'erreurs consécutives
    Cette fonction utilise Application Insights pour déclencher l'alerte
    """
    try:
        alert_properties = {
            "alertType": "ConsecutiveErrors",
            "errorCount": str(ERROR_THRESHOLD),
            "timeWindow": str(ERROR_WINDOW),
            "notificationEmail": NOTIFICATION_EMAIL
        }
        
        if properties:
            alert_properties.update(properties)
            
        # Envoi d'un événement spécial qui déclenchera l'alerte configurée
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
                    "name": "ConsecutiveModelErrors",
                    "properties": alert_properties,
                    "measurements": {
                        "errorCount": ERROR_THRESHOLD
                    }
                }
            }
        }
        
        response = requests.post(APPINSIGHTS_ENDPOINT, json=payload)
        
        if response.status_code != 200:
            logger.warning(f"Erreur lors de l'envoi de l'alerte à Application Insights: {response.status_code} - {response.text}")
            return False
            
        logger.warning(f"ALERTE: {ERROR_THRESHOLD} erreurs consécutives détectées. Notification envoyée.")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi de l'alerte: {str(e)}")
        return False

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
        # Suivi des erreurs consécutives
        if is_incorrect:
            with error_lock:
                error_history.append(time.time())
                
            # Vérifier s'il y a des erreurs consécutives
            if check_consecutive_errors():
                send_alert_email(properties)
        else:
            # Réinitialiser le compteur d'erreurs en cas de prédiction correcte
            with error_lock:
                error_history.clear()
        
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
                        "processingTime": 0.0,
                        "isError": 1.0 if is_incorrect else 0.0
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