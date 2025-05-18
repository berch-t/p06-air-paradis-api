import os
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import traceback
import logging
from datetime import datetime
import requests

# Configuration de la journalisation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration pour Azure Application Insights
APPINSIGHTS_INSTRUMENTATION_KEY = os.getenv('APPINSIGHTS_INSTRUMENTATION_KEY', '')
APPINSIGHTS_ENDPOINT = "https://dc.services.visualstudio.com/v2/track"

app = Flask(__name__)

# Fonction pour prétraiter un tweet
def preprocess_tweet(tweet):
    """
    Prétraitement d'un tweet pour la prédiction
    """
    # Conversion en minuscules
    tweet = tweet.lower()
    
    # Suppression des mentions
    tweet = re.sub(r'@\w+', '', tweet)
    
    # Suppression des URLs
    tweet = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', tweet)
    
    # Suppression des hashtags (on garde le mot sans #)
    tweet = re.sub(r'#(\w+)', r'\1', tweet)
    
    # Suppression des caractères spéciaux et de la ponctuation
    tweet = re.sub(r'[^\w\s]', '', tweet)
    
    # Suppression des chiffres
    tweet = re.sub(r'\d+', '', tweet)
    
    return tweet

# Fonction pour charger le modèle et le tokenizer
def load_sentiment_model():
    """
    Charge le modèle et le tokenizer pour l'analyse de sentiment
    """
    try:
        # Chargement du modèle
        model_path = os.path.join('models', 'best_advanced_model_BiLSTM_Word2Vec.h5')
        model = load_model(model_path)
        logger.info(f"Modèle chargé depuis {model_path}")
        
        # Chargement du tokenizer
        tokenizer_path = os.path.join('models', 'tokenizer.pickle')
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        logger.info(f"Tokenizer chargé depuis {tokenizer_path}")
        
        # Chargement de la configuration
        config_path = os.path.join('models', 'model_config.pickle')
        with open(config_path, 'rb') as handle:
            config = pickle.load(handle)
        logger.info(f"Configuration chargée depuis {config_path}")
        
        return model, tokenizer, config
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# Fonction pour envoyer des traces à Azure Application Insights
def send_to_appinsights(tweet, prediction, is_incorrect=False):
    """
    Envoie des données de télémétrie à Azure Application Insights
    """
    if not APPINSIGHTS_INSTRUMENTATION_KEY:
        logger.warning("Clé d'instrumentation Azure Application Insights non configurée.")
        return
    
    try:
        # Création du payload pour Application Insights
        payload = {
            "name": "Microsoft.ApplicationInsights.Event",
            "time": datetime.utcnow().isoformat() + "Z",
            "iKey": APPINSIGHTS_INSTRUMENTATION_KEY,
            "tags": {
                "ai.cloud.roleInstance": "sentiment-analysis-api"
            },
            "data": {
                "baseType": "EventData",
                "baseData": {
                    "ver": 2,
                    "name": "IncorrectPrediction" if is_incorrect else "Prediction",
                    "properties": {
                        "tweet": tweet,
                        "predicted_sentiment": prediction,
                        "is_incorrect": str(is_incorrect)
                    }
                }
            }
        }
        
        # Envoi de la télémétrie
        response = requests.post(APPINSIGHTS_ENDPOINT, json=payload)
        if response.status_code != 200:
            logger.warning(f"Erreur lors de l'envoi à Application Insights: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi à Application Insights: {str(e)}")

# Initialisation du modèle et du tokenizer
model, tokenizer, config = load_sentiment_model()
MAX_SEQUENCE_LENGTH = config.get('max_sequence_length', 50)

@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint de vérification de l'état de l'API
    """
    return jsonify({
        "status": "healthy",
        "model": "sentiment_analysis",
        "version": "1.0.0"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint pour la prédiction du sentiment d'un tweet
    """
    try:
        # Récupération des données
        data = request.get_json(force=True)
        tweet = data.get('tweet', '')
        
        if not tweet:
            return jsonify({
                "error": "Le champ 'tweet' est obligatoire"
            }), 400
        
        # Prétraitement du tweet
        processed_tweet = preprocess_tweet(tweet)
        
        # Tokenisation et padding
        sequence = tokenizer.texts_to_sequences([processed_tweet])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
        
        # Prédiction
        prediction = model.predict(padded_sequence)[0][0]
        sentiment = "Positif" if prediction > 0.5 else "Négatif"
        confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)
        
        # Envoi de la télémétrie
        send_to_appinsights(tweet, sentiment)
        
        # Retour de la prédiction
        return jsonify({
            "tweet": tweet,
            "processed_tweet": processed_tweet,
            "sentiment": sentiment,
            "confidence": confidence
        })
    
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Endpoint pour recevoir le feedback sur une prédiction
    """
    try:
        # Récupération des données
        data = request.get_json(force=True)
        tweet = data.get('tweet', '')
        predicted_sentiment = data.get('predicted_sentiment', '')
        is_correct = data.get('is_correct', True)
        
        if not tweet or not predicted_sentiment:
            return jsonify({
                "error": "Les champs 'tweet' et 'predicted_sentiment' sont obligatoires"
            }), 400
        
        # Si la prédiction est incorrecte, envoi de la trace à Application Insights
        if not is_correct:
            send_to_appinsights(tweet, predicted_sentiment, is_incorrect=True)
            logger.info(f"Feedback négatif reçu pour le tweet: {tweet}")
        
        return jsonify({
            "status": "success",
            "message": "Feedback enregistré avec succès"
        })
    
    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement du feedback: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == '__main__':
    # Pour le développement local
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)