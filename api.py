import os
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
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
APPINSIGHTS_INSTRUMENTATION_KEY = os.getenv('APPINSIGHTS_INSTRUMENTATION_KEY', '9af56b4d-4ad5-4643-ba29-41d154893ad4')
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

# Fonction pour charger le modèle DistilBERT et le tokenizer
def load_bert_model():
    """
    Charge le modèle DistilBERT et le tokenizer pour l'analyse de sentiment
    """
    try:
        # Définition des chemins
        model_path = os.path.join('models', 'bert', 'best_model_bert')
        tokenizer_path = os.path.join('models', 'bert', 'tokenizer_bert')
        
        # Chargement du modèle
        model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
        logger.info(f"Modèle BERT chargé depuis {model_path}")
        
        # Chargement du tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
        logger.info(f"Tokenizer BERT chargé depuis {tokenizer_path}")
        
        # Chargement de la configuration
        config = {'max_sequence_length': 64}
        logger.info("Configuration BERT chargée")
        
        return model, tokenizer, config
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle BERT: {str(e)}")
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
try:
    model, tokenizer, config = load_bert_model()
    MAX_SEQUENCE_LENGTH = config.get('max_sequence_length', 64)
    logger.info(f"Modèle BERT initialisé avec sequence_length={MAX_SEQUENCE_LENGTH}")
except Exception as e:
    logger.error(f"Erreur lors de l'initialisation du modèle: {str(e)}")
    logger.error("Le serveur va démarrer mais les prédictions ne seront pas disponibles jusqu'à ce que le modèle soit chargé correctement.")
    model = None
    tokenizer = None
    MAX_SEQUENCE_LENGTH = 64

@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint de vérification de l'état de l'API
    """
    status = "healthy" if model is not None and tokenizer is not None else "degraded"
    return jsonify({
        "status": status,
        "model": "BERT_sentiment_analysis",
        "version": "1.0.0"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint pour la prédiction du sentiment d'un tweet
    """
    try:
        # Vérification que le modèle est chargé
        if model is None or tokenizer is None:
            return jsonify({
                "error": "Le modèle n'est pas initialisé correctement"
            }), 503
        
        # Récupération des données
        data = request.get_json(force=True)
        tweet = data.get('tweet', '')
        
        if not tweet:
            return jsonify({
                "error": "Le champ 'tweet' est obligatoire"
            }), 400
        
        # Prétraitement du tweet
        processed_tweet = preprocess_tweet(tweet)
        
        # Tokenisation
        inputs = tokenizer(
            processed_tweet,
            add_special_tokens=True,
            max_length=MAX_SEQUENCE_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        
        # Prédiction
        outputs = model(inputs)
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]
        predicted_class = np.argmax(probabilities)
        
        # Détermination du sentiment
        sentiment = "Positif" if predicted_class == 1 else "Négatif"
        confidence = float(probabilities[predicted_class])
        
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