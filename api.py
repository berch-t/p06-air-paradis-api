import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory, render_template_string
import re
import traceback
import logging
from datetime import datetime
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import json
import time
from transformers import BertTokenizer, TFBertForSequenceClassification
import nltk
from nltk.tokenize import word_tokenize
import io
import base64
from PIL import Image
import sys
from pathlib import Path

# Configuration du logging
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

# Vérifier et créer le dossier 'models' s'il n'existe pas
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('models/bert'):
    os.makedirs('models/bert')

# Téléchargement des ressources NLTK nécessaires
try:
    nltk.download('punkt', quiet=True)
except:
    pass

# Chargement du modèle BERT
def load_bert_model():
    """
    Charge le modèle BERT pour l'analyse de sentiment
    """
    try:
        # Si le modèle est déjà téléchargé et sauvegardé localement
        if os.path.exists('models/bert/tokenizer') and os.path.exists('models/bert/model'):
            tokenizer = BertTokenizer.from_pretrained('models/bert/tokenizer')
            model = TFBertForSequenceClassification.from_pretrained('models/bert/model')
            logger.info("Modèle BERT chargé depuis le cache local")
        else:
            # Téléchargement du modèle depuis Hugging Face
            logger.info("Téléchargement du modèle BERT depuis Hugging Face")
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
            
            # Sauvegarde du modèle localement pour une utilisation future
            tokenizer.save_pretrained('models/bert/tokenizer')
            model.save_pretrained('models/bert/model')
            
        return model, tokenizer
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle BERT: {str(e)}")
        logger.error(traceback.format_exc())
        raise

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

# Fonction pour prédire le sentiment avec BERT
def predict_sentiment_bert(tweet, max_length=128):
    """
    Prédit le sentiment d'un tweet en utilisant BERT
    """
    try:
        # Prétraitement du tweet
        processed_tweet = preprocess_tweet(tweet)
        
        # Chargement du modèle (avec mise en cache)
        model, tokenizer = load_bert_model()
        
        # Tokenisation
        encoding = tokenizer.encode_plus(
            processed_tweet,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='tf'
        )
        
        # Préparation des entrées
        inputs = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'token_type_ids': encoding['token_type_ids']
        }
        
        # Prédiction
        outputs = model(inputs)
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]
        
        # La classe 1 représente le sentiment positif, la classe 0 le sentiment négatif
        sentiment = "Positif" if probabilities[1] > probabilities[0] else "Négatif"
        confidence = float(probabilities[1] if sentiment == "Positif" else probabilities[0])
        
        # Création de la réponse
        result = {
            'tweet': tweet,
            'processed_tweet': processed_tweet,
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                'positive': float(probabilities[1]),
                'negative': float(probabilities[0])
            }
        }
        
        return result
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction avec BERT: {str(e)}")
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

# Génération d'un jeu de données d'exemple pour les visualisations
def generate_sample_data(size=100):
    """
    Génère un jeu de données d'exemple pour les visualisations
    """
    np.random.seed(42)
    tweets = [
        "I love flying with Air Paradis! Best service ever!",
        "The staff was so friendly and helpful",
        "Comfortable seats and delicious food",
        "On-time departure and arrival, very professional",
        "Amazing experience from check-in to landing",
        "Terrible service, never flying with them again",
        "My flight was delayed for hours with no explanation",
        "Lost my luggage and no compensation offered",
        "The seats were uncomfortable and the food was cold",
        "Rude staff and chaotic boarding process"
    ]
    
    data = []
    for _ in range(size):
        idx = np.random.randint(0, len(tweets))
        tweet = tweets[idx]
        result = predict_sentiment_bert(tweet)
        
        # Ajouter une date aléatoire dans les 30 derniers jours
        days_ago = np.random.randint(1, 31)
        date = (datetime.now() - pd.Timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        data.append({
            'tweet': tweet,
            'sentiment': result['sentiment'],
            'confidence': result['confidence'],
            'date': date
        })
    
    return pd.DataFrame(data)

# Routes API Flask

@app.route('/favicon.ico')
def favicon():
    """Endpoint pour servir le favicon"""
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint de vérification de l'état de l'API
    """
    return jsonify({
        "status": "healthy",
        "model": "sentiment_analysis_bert",
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
        
        # Prédiction avec BERT
        result = predict_sentiment_bert(tweet)
        
        # Envoi de la télémétrie
        send_to_appinsights(tweet, result['sentiment'])
        
        # Retour de la prédiction
        return jsonify(result)
    
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

# HTML template pour l'interface utilisateur Streamlit intégrée
STREAMLIT_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Air Paradis - Analyse de Sentiment</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f7fa;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            text-align: center;
        }
        h1 {
            color: #003366;
            margin-bottom: 20px;
        }
        iframe {
            width: 100%;
            height: 800px;
            border: none;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>✈️ Air Paradis - Analyse de Sentiment</h1>
        <iframe src="http://localhost:8501"></iframe>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    """
    Page d'accueil avec interface utilisateur intégrée
    """
    return render_template_string(STREAMLIT_HTML)

# Application Streamlit (exécutée séparément)
def run_streamlit_app():
    """
    Application Streamlit pour l'analyse de sentiment des tweets
    """
    # Configuration de la page
    st.set_page_config(
        page_title="Air Paradis - Analyse de Sentiment",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Application CSS personnalisé
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Montserrat', sans-serif;
        }
        
        .main-header {
            font-size: 2.8rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 2rem;
            text-align: center;
            text-shadow: 0px 2px 4px rgba(0,0,0,0.1);
            background: linear-gradient(90deg, #1E3A8A, #3B82F6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }
        
        .sub-header {
            font-size: 1.8rem;
            font-weight: 600;
            color: #2563EB;
            margin-top: 2rem;
            margin-bottom: 1rem;
            border-bottom: 2px solid #3B82F6;
            padding-bottom: 0.5rem;
        }
        
        .card {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
            margin-bottom: 25px;
            border: 1px solid #E5E7EB;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
        }
        
        .sentiment-positive {
            color: #059669;
            font-weight: 600;
            font-size: 1.3rem;
            background: rgba(5, 150, 105, 0.1);
            padding: 8px 16px;
            border-radius: 8px;
            display: inline-block;
        }
        
        .sentiment-negative {
            color: #DC2626;
            font-weight: 600;
            font-size: 1.3rem;
            background: rgba(220, 38, 38, 0.1);
            padding: 8px 16px;
            border-radius: 8px;
            display: inline-block;
        }
        
        .confidence-meter {
            height: 12px;
            background-color: #E5E7EB;
            border-radius: 6px;
            margin: 10px 0;
            overflow: hidden;
        }
        
        .confidence-value {
            height: 100%;
            border-radius: 6px;
            transition: width 0.5s ease;
        }
        
        .confidence-high {
            background: linear-gradient(90deg, #34D399, #059669);
        }
        
        .confidence-medium {
            background: linear-gradient(90deg, #FBBF24, #D97706);
        }
        
        .confidence-low {
            background: linear-gradient(90deg, #F87171, #DC2626);
        }
        
        .btn-primary {
            background-color: #3B82F6;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: 600;
            transition: background-color 0.3s ease;
            border: none;
            cursor: pointer;
        }
        
        .btn-primary:hover {
            background-color: #2563EB;
        }
        
        .footer {
            text-align: center;
            margin-top: 3rem;
            color: #6B7280;
            font-size: 0.9rem;
            padding: 20px;
            border-top: 1px solid #E5E7EB;
        }
        
        .stats-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            text-align: center;
            border: 1px solid #E5E7EB;
        }
        
        .stats-number {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 0;
        }
        
        .stats-label {
            font-size: 1rem;
            color: #6B7280;
        }
        
        /* Animation de chargement */
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        
        .loading-animation {
            animation: pulse 1.5s infinite;
        }
        
        /* Feedback buttons */
        .feedback-button-correct {
            background-color: #059669;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .feedback-button-correct:hover {
            background-color: #047857;
            transform: translateY(-2px);
        }
        
        .feedback-button-incorrect {
            background-color: #DC2626;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .feedback-button-incorrect:hover {
            background-color: #B91C1C;
            transform: translateY(-2px);
        }
        
        /* Input de texte stylisé */
        .tweet-input {
            border: 2px solid #E5E7EB;
            border-radius: 12px;
            padding: 15px;
            font-size: 16px;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        .tweet-input:focus {
            border-color: #3B82F6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3);
        }
        
        /* Fading animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-fade-in {
            animation: fadeIn 0.5s ease-out forwards;
        }
    </style>
    """, unsafe_allow_html=True)

    # En-tête
    st.markdown('<h1 class="main-header">✈️ Air Paradis - Analyse de Sentiment</h1>', unsafe_allow_html=True)

    # Barre latérale
    st.sidebar.image("https://i.ibb.co/pdjJzvD/air-paradis-logo.png", width=200)
    st.sidebar.markdown("## À propos")
    st.sidebar.info(
        "Cette application utilise un modèle BERT pour analyser le sentiment des tweets concernant la compagnie aérienne Air Paradis. "
        "Elle permet de déterminer si un tweet exprime un sentiment positif ou négatif, avec un score de confiance."
    )

    st.sidebar.markdown("## Comment ça marche")
    st.sidebar.markdown(
        "1. Entrez un tweet dans le champ de texte\n"
        "2. Cliquez sur 'Analyser le sentiment'\n"
        "3. Consultez les résultats de l'analyse\n"
        "4. Donnez votre feedback sur la qualité de la prédiction"
    )

    st.sidebar.markdown("## Statistiques")
    if 'predictions_history' not in st.session_state:
        st.session_state.predictions_history = []

    total_predictions = len(st.session_state.predictions_history)
    
    # Création de statistiques pour la démo, même sans historique
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
        st.markdown(f'<p class="stats-number">{total_predictions}</p>', unsafe_allow_html=True)
        st.markdown('<p class="stats-label">Analyses</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        positive_count = sum(1 for p in st.session_state.predictions_history if p.get('sentiment') == 'Positif')
        positive_percent = (positive_count / total_predictions * 100) if total_predictions > 0 else 0
        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
        st.markdown(f'<p class="stats-number">{positive_percent:.1f}%</p>', unsafe_allow_html=True)
        st.markdown('<p class="stats-label">Positifs</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        correct_predictions = sum(1 for p in st.session_state.predictions_history if p.get('feedback', True))
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        st.markdown('<div class="stats-card">', unsafe_allow_html=True)
        st.markdown(f'<p class="stats-number">{accuracy:.1f}%</p>', unsafe_allow_html=True)
        st.markdown('<p class="stats-label">Précision</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Zone principale
    st.markdown('<h2 class="sub-header">Analyse d\'un nouveau tweet</h2>', unsafe_allow_html=True)

    # Entrée du tweet avec placeholder
    tweet_input = st.text_area(
        "Entrez un tweet à analyser",
        placeholder="Exemple : J'ai adoré mon vol avec Air Paradis ! Le service était impeccable et le personnel très attentionné.",
        height=100,
        key="tweet_input",
        help="Entrez ici le texte du tweet que vous souhaitez analyser"
    )

    # Analyse du sentiment
    if st.button("Analyser le sentiment", key="analyze_button", type="primary"):
        if tweet_input:
            # Afficher animation de chargement
            with st.spinner("Analyse en cours..."):
                # Simulation d'un léger délai pour montrer l'animation
                time.sleep(0.5)
                
                # Appel à l'API pour la prédiction (ici on utilise directement la fonction)
                result = predict_sentiment_bert(tweet_input)
                
                # Stocker les résultats dans la session
                result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.predictions_history.append(result)
                
                # Affichage des résultats
                sentiment = result.get('sentiment')
                confidence = result.get('confidence', 0) * 100
                
                # Définition des classes CSS en fonction du sentiment et de la confiance
                sentiment_class = "sentiment-positive" if sentiment == "Positif" else "sentiment-negative"
                
                if confidence >= 80:
                    confidence_class = "confidence-high"
                elif confidence >= 60:
                    confidence_class = "confidence-medium"
                else:
                    confidence_class = "confidence-low"
                
                # Création d'une carte pour les résultats
                st.markdown('<div class="card animate-fade-in">', unsafe_allow_html=True)
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Tweet analysé :** {tweet_input}", unsafe_allow_html=True)
                    st.markdown(f"**Tweet prétraité :** {result.get('processed_tweet', '')}", unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"**Sentiment détecté :** <span class='{sentiment_class}'>{sentiment}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Niveau de confiance :** {confidence:.1f}%", unsafe_allow_html=True)
                    st.markdown(f'<div class="confidence-meter"><div class="confidence-value {confidence_class}" style="width: {confidence}%;"></div></div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Visualisation des probabilités
                fig = go.Figure(go.Bar(
                    x=[result['probabilities']['positive'], result['probabilities']['negative']],
                    y=['Positif', 'Négatif'],
                    orientation='h',
                    marker=dict(
                        color=['#059669', '#DC2626'],
                        line=dict(color='rgba(0, 0, 0, 0)', width=1)
                    )
                ))
                
                fig.update_layout(
                    title='Probabilités par classe',
                    xaxis_title='Probabilité',
                    yaxis_title='Sentiment',
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Montserrat, sans-serif")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Demande de feedback
                st.markdown('<h3 style="font-size:1.3rem;">Êtes-vous d\'accord avec cette prédiction ?</h3>', unsafe_allow_html=True)
                
                feedback_col1, feedback_col2 = st.columns(2)
                
                with feedback_col1:
                    if st.button("✅ Oui, c'est correct", key="feedback_correct", type="primary"):
                        # Mise à jour du feedback dans l'historique
                        st.session_state.predictions_history[-1]['feedback'] = True
                        st.success("Merci pour votre feedback !")
                
                with feedback_col2:
                    if st.button("❌ Non, c'est incorrect", key="feedback_incorrect", type="secondary"):
                        # Mise à jour du feedback dans l'historique
                        st.session_state.predictions_history[-1]['feedback'] = False
                        st.warning("Merci pour votre feedback ! Nous utiliserons cette information pour améliorer notre modèle.")
        else:
            st.warning("Veuillez entrer un tweet à analyser.")

    # Historique des prédictions
    if st.session_state.predictions_history:
        st.markdown('<h2 class="sub-header">Historique des prédictions</h2>', unsafe_allow_html=True)
        
        # Création d'un DataFrame pour l'historique
        history_df = pd.DataFrame(st.session_state.predictions_history)
        
        # Affichage de l'historique
        with st.expander("Voir l'historique complet des prédictions", expanded=False):
            # Formater les données pour l'affichage
            if 'probabilities' in history_df.columns:
                history_df = history_df.drop(columns=['probabilities'])
            
            # Conversion des valeurs de confiance en pourcentage
            if 'confidence' in history_df.columns:
                history_df['confidence'] = history_df['confidence'].apply(lambda x: f"{x*100:.1f}%" if isinstance(x, (float, int)) else x)
            
            st.dataframe(
                history_df[['tweet', 'sentiment', 'confidence', 'feedback', 'timestamp']].rename(
                    columns={
                        'tweet': 'Tweet',
                        'sentiment': 'Sentiment',
                        'confidence': 'Confiance',
                        'feedback': 'Feedback Correct',
                        'timestamp': 'Horodatage'
                    }
                ),
                use_container_width=True
            )
        
        # Visualisations
        st.markdown('<h2 class="sub-header">Visualisations</h2>', unsafe_allow_html=True)
        
        # Si on n'a pas assez de données dans l'historique, on génère des données d'exemple
        if len(st.session_state.predictions_history) < 10:
            with st.expander("Générer des données de démonstration pour les visualisations", expanded=False):
                if st.button("Générer 100 prédictions aléatoires", key="generate_data"):
                    sample_df = generate_sample_data(100)
                    
                    # Ajouter à l'historique existant
                    for _, row in sample_df.iterrows():
                        st.session_state.predictions_history.append({
                            'tweet': row['tweet'],
                            'sentiment': row['sentiment'],
                            'confidence': row['confidence'],
                            'feedback': True,  # On suppose que toutes les prédictions générées sont correctes
                            'timestamp': row['date']
                        })
                    
                    st.success("Données de démonstration générées avec succès !")
        
        # Création du DataFrame pour les visualisations
        viz_df = pd.DataFrame(st.session_state.predictions_history)
        
        # Si nous avons des données, afficher les visualisations
        if not viz_df.empty:
            # Ajouter une colonne de date pour les visualisations temporelles
            if 'timestamp' in viz_df.columns:
                viz_df['date'] = pd.to_datetime(viz_df['timestamp']).dt.date
            
            # Visualisations en 2 colonnes
            col1, col2 = st.columns(2)
            
            with col1:
                # Répartition des sentiments
                sentiment_counts = viz_df['sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                
                fig = px.pie(
                    sentiment_counts, 
                    values='Count', 
                    names='Sentiment', 
                    title='Répartition des sentiments',
                    color='Sentiment',
                    color_discrete_map={'Positif': '#059669', 'Négatif': '#DC2626'},
                    hole=0.4
                )
                
                fig.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Montserrat, sans-serif")
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Précision du modèle
                if 'feedback' in viz_df.columns:
                    feedback_counts = viz_df['feedback'].fillna(True).value_counts().reset_index()
                    feedback_counts.columns = ['Correct', 'Count']
                    
                    fig = px.pie(
                        feedback_counts, 
                        values='Count', 
                        names='Correct', 
                        title='Précision du modèle (selon le feedback)',
                        color='Correct',
                        color_discrete_map={True: '#059669', False: '#DC2626'},
                        hole=0.4
                    )
                    
                    fig.update_layout(
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                        margin=dict(l=20, r=20, t=50, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Montserrat, sans-serif")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Distribution temporelle des sentiments
            if 'date' in viz_df.columns and len(viz_df['date'].unique()) > 1:
                st.markdown("### Évolution temporelle des sentiments")
                
                # Agrégation par date et sentiment
                time_sentiment = viz_df.groupby(['date', 'sentiment']).size().reset_index(name='count')
                
                # Graphique temporel
                fig = px.line(
                    time_sentiment, 
                    x='date', 
                    y='count', 
                    color='sentiment',
                    title='Évolution des sentiments au fil du temps',
                    color_discrete_map={'Positif': '#059669', 'Négatif': '#DC2626'}
                )
                
                fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Nombre de tweets',
                    legend_title='Sentiment',
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Montserrat, sans-serif")
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            # Carte thermique de la confiance par sentiment
            st.markdown("### Distribution des scores de confiance")
            
            # Création de buckets pour les niveaux de confiance
            if 'confidence' in viz_df.columns:
                # S'assurer que confidence est numérique
                viz_df['confidence_num'] = pd.to_numeric(viz_df['confidence'], errors='coerce')
                viz_df['confidence_bucket'] = pd.cut(
                    viz_df['confidence_num'], 
                    bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0],
                    labels=['<60%', '60-70%', '70-80%', '80-90%', '90-100%']
                )
                
                # Compte par bucket et sentiment
                confidence_heatmap = viz_df.groupby(['confidence_bucket', 'sentiment']).size().reset_index(name='count')
                confidence_pivot = confidence_heatmap.pivot(index='confidence_bucket', columns='sentiment', values='count').fillna(0)
                
                # Création de la heatmap
                fig = px.imshow(
                    confidence_pivot,
                    labels=dict(x="Sentiment", y="Score de confiance", color="Nombre de tweets"),
                    x=confidence_pivot.columns,
                    y=confidence_pivot.index,
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    title='Distribution des scores de confiance par sentiment',
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Montserrat, sans-serif")
                )
                
                st.plotly_chart(fig, use_container_width=True)

    # Pied de page
    st.markdown('<div class="footer">© 2025 Air Paradis - Analyse de Sentiment | Développé par MIC (Marketing Intelligence Consulting)</div>', 
                unsafe_allow_html=True)

# Point d'entrée principal
if __name__ == '__main__':
    # Vérifier si nous sommes dans Streamlit ou Flask
    if 'STREAMLIT_RUN_MODE' in os.environ:
        run_streamlit_app()
    else:
        # Si nous sommes sur Azure ou un autre environnement de production
        port = int(os.environ.get('PORT', 5000))
        
        # Démarrer le serveur Flask
        app.run(host='0.0.0.0', port=port, debug=False)