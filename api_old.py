"""
Version simplifiée de l'API pour l'analyse de sentiment
Cette version réduit la complexité et corrige les problèmes de chargement indéfini
"""
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
import plotly.express as px
import plotly.graph_objects as go
import time
import nltk
import sys
import pickle

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

# Téléchargement des ressources NLTK nécessaires
try:
    nltk.download('punkt', quiet=True)
except:
    pass

# Chargement du modèle BiLSTM avec Word2Vec
def load_model():
    """Charge le modèle BiLSTM avec Word2Vec pour l'analyse de sentiment"""
    try:
        # Chemins vers le modèle et le tokenizer
        model_path = 'models/best_advanced_model_BiLSTM_Word2Vec.h5'
        tokenizer_path = 'models/tokenizer.pickle'
        
        # Vérifier si les chemins existent
        if not os.path.exists(model_path):
            logger.error(f"Modèle BiLSTM non trouvé. Vérifiez que le fichier existe: {model_path}")
            raise FileNotFoundError(f"Modèle BiLSTM non trouvé au chemin: {model_path}")
        
        if not os.path.exists(tokenizer_path):
            logger.error(f"Tokenizer non trouvé. Vérifiez que le fichier existe: {tokenizer_path}")
            raise FileNotFoundError(f"Tokenizer non trouvé au chemin: {tokenizer_path}")
        
        # Charger le tokenizer
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        # Charger le modèle
        model = tf.keras.models.load_model(model_path)
        
        logger.info("Modèle BiLSTM et tokenizer chargés avec succès")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle : {str(e)}")
        logger.error(traceback.format_exc())
        raise

# Fonction pour prétraiter un tweet
def preprocess_tweet(tweet):
    """Prétraitement d'un tweet pour la prédiction"""
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

# Fonction de prédiction du sentiment
def predict_sentiment(tweet, max_length=50):
    """Prédit le sentiment d'un tweet en utilisant le modèle BiLSTM avec Word2Vec"""
    try:
        # Prétraitement du tweet
        processed_tweet = preprocess_tweet(tweet)
        
        # Chargement du modèle et du tokenizer
        model, tokenizer = load_model()
        
        # Tokenisation
        sequences = tokenizer.texts_to_sequences([processed_tweet])
        
        # Padding
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences,
            maxlen=max_length,
            padding='post'
        )
        
        # Prédiction
        prediction = model.predict(padded_sequences)
        probability = prediction[0][0]
        
        # Journalisation
        logger.info(f"Tweet brut: {tweet}")
        logger.info(f"Tweet prétraité: {processed_tweet}")
        logger.info(f"Probabilité positive: {probability:.4f}")
        
        # Décision avec seuil standard
        threshold = 0.5
        sentiment = "Positif" if probability > threshold else "Négatif"
        confidence = float(probability if sentiment == "Positif" else 1 - probability)
        
        # Création de la réponse
        result = {
            'tweet': tweet,
            'processed_tweet': processed_tweet,
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                'positive': float(probability),
                'negative': float(1 - probability)
            }
        }
        
        return result
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Erreur lors du chargement du modèle: {str(e)}")
        return None

# Génération de données d'exemple simples
def generate_sample_data(size=20):
    """Génère un jeu de données d'exemple pour les visualisations"""
    np.random.seed(42)
    tweets = [
        "I love flying with Air Paradis! Best service ever!",
        "The staff was so friendly and helpful",
        "Comfortable seats and delicious food",
        "Terrible service, never flying with them again",
        "My flight was delayed for hours with no explanation",
        "Lost my luggage and no compensation offered"
    ]
    
    data = []
    for _ in range(size):
        idx = np.random.randint(0, len(tweets))
        tweet = tweets[idx]
        
        # Sentiment direct sans appeler le modèle pour éviter les erreurs
        sentiment = "Positif" if idx < 3 else "Négatif"
        confidence = np.random.uniform(0.7, 0.95)
        
        # Date aléatoire dans les 30 derniers jours
        days_ago = np.random.randint(1, 31)
        date = (datetime.now() - pd.Timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        data.append({
            'tweet': tweet,
            'sentiment': sentiment,
            'confidence': confidence,
            'date': date
        })
    
    return pd.DataFrame(data)

# Routes API Flask
@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de vérification de l'état de l'API"""
    return jsonify({
        "status": "healthy",
        "model": "sentiment_analysis_bilstm_word2vec",
        "version": "1.0.0"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint pour la prédiction du sentiment d'un tweet"""
    try:
        data = request.get_json(force=True)
        tweet = data.get('tweet', '')
        
        if not tweet:
            return jsonify({"error": "Le champ 'tweet' est obligatoire"}), 400
        
        result = predict_sentiment(tweet)
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Afficher tous les arguments pour le débogage
print(f"Current working directory: {os.getcwd()}")
print(f"Python executable: {sys.executable}")
print(f"Script path: {__file__}")
print(f"Command line arguments: {sys.argv}")

# Application Streamlit
def main():
    """Application Streamlit pour l'analyse de sentiment des tweets"""
    # Configuration de la page
    st.set_page_config(
        page_title="Air Paradis - Analyse de Sentiment",
        page_icon="✈️",
        layout="wide"
    )

    # Initialisation des variables de session
    if 'predictions_history' not in st.session_state:
        st.session_state.predictions_history = []

    # CSS minimal
    st.markdown("""
    <style>
    .header {
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .positive {
        color: #059669;
        font-weight: bold;
    }
    .negative {
        color: #DC2626;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    # En-tête
    st.markdown('<h1 class="header">✈️ Air Paradis - Analyse de Sentiment</h1>', unsafe_allow_html=True)

    # Interface d'analyse
    st.subheader("Analyse d'un nouveau tweet")
    tweet_input = st.text_area("Entrez un tweet à analyser", height=100)

    # Analyse du sentiment
    if st.button("Analyser le sentiment", type="primary"):
        if tweet_input:
            with st.spinner("Analyse en cours..."):
                try:
                    # Prédiction de démonstration
                    result = predict_demo(tweet_input)
                    
                    # Stockage dans l'historique
                    result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.predictions_history.append(result)
                    
                    # Affichage des résultats
                    sentiment = result['sentiment']
                    confidence = result['confidence'] * 100
                    
                    st.write(f"**Tweet:** {tweet_input}")
                    st.write(f"**Tweet prétraité:** {result['processed_tweet']}")
                    
                    if sentiment == "Positif":
                        st.markdown(f"**Sentiment:** <span class='positive'>{sentiment}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**Sentiment:** <span class='negative'>{sentiment}</span>", unsafe_allow_html=True)
                    
                    st.write(f"**Confiance:** {confidence:.1f}%")
                    
                    # Visualisation des probabilités
                    fig = go.Figure(go.Bar(
                        x=[result['probabilities']['positive'], result['probabilities']['negative']],
                        y=['Positif', 'Négatif'],
                        orientation='h',
                        marker=dict(color=['#059669', '#DC2626'])
                    ))
                    
                    fig.update_layout(
                        title='Probabilités par classe',
                        xaxis_title='Probabilité',
                        yaxis_title='Sentiment',
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feedback
                    st.subheader("Êtes-vous d'accord avec cette prédiction?")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("✅ Oui, c'est correct"):
                            st.session_state.predictions_history[-1]['feedback'] = True
                            st.success("Merci pour votre feedback!")
                    
                    with col2:
                        if st.button("❌ Non, c'est incorrect"):
                            st.session_state.predictions_history[-1]['feedback'] = False
                            st.warning("Merci pour votre feedback! Nous utiliserons cette information pour améliorer notre modèle.")
                
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse: {str(e)}")
        else:
            st.warning("Veuillez entrer un tweet à analyser.")

    # Historique et visualisations
    if st.session_state.predictions_history:
        st.subheader("Historique des prédictions")
        
        # Création du DataFrame
        history_df = pd.DataFrame(st.session_state.predictions_history)
        
        # Affichage de l'historique
        with st.expander("Voir l'historique"):
            # Préparation des données pour l'affichage
            display_df = history_df.copy()
            
            if 'probabilities' in display_df.columns:
                display_df = display_df.drop(columns=['probabilities'])
            
            if 'confidence' in display_df.columns:
                display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.1f}%" if isinstance(x, (float, int)) else x)
            
            # Assurer que la colonne feedback existe
            if 'feedback' not in display_df.columns:
                display_df['feedback'] = None
            
            # Affichage
            cols_to_display = ['tweet', 'sentiment', 'confidence', 'feedback', 'timestamp']
            cols_to_display = [col for col in cols_to_display if col in display_df.columns]
            
            rename_dict = {
                'tweet': 'Tweet',
                'sentiment': 'Sentiment',
                'confidence': 'Confiance',
                'feedback': 'Feedback Correct',
                'timestamp': 'Horodatage'
            }
            
            st.dataframe(
                display_df[cols_to_display].rename(columns=rename_dict),
                use_container_width=True
            )
        
        # Visualisations basiques
        st.subheader("Visualisations")
        
        # Option pour générer des données d'exemple
        if len(st.session_state.predictions_history) < 5:
            st.info("Pas assez de données pour des visualisations pertinentes.")
            if st.button("Générer des données d'exemple"):
                sample_df = generate_sample_data(20)
                
                for _, row in sample_df.iterrows():
                    st.session_state.predictions_history.append({
                        'tweet': row['tweet'],
                        'sentiment': row['sentiment'],
                        'confidence': row['confidence'],
                        'feedback': True,
                        'timestamp': row['date']
                    })
                
                st.success("Données générées avec succès!")
                st.experimental_rerun()
        else:
            # Visualisations simples
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution des sentiments
                sentiment_counts = history_df['sentiment'].value_counts()
                fig = px.pie(
                    names=sentiment_counts.index,
                    values=sentiment_counts.values,
                    title="Distribution des sentiments",
                    color=sentiment_counts.index,
                    color_discrete_map={'Positif': '#059669', 'Négatif': '#DC2626'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Distribution des confiances
                if 'confidence' in history_df.columns:
                    fig = px.histogram(
                        history_df,
                        x='confidence',
                        color='sentiment',
                        title="Distribution des scores de confiance",
                        color_discrete_map={'Positif': '#059669', 'Négatif': '#DC2626'},
                        nbins=10
                    )
                    st.plotly_chart(fig, use_container_width=True)

# Point d'entrée principal - exécute directement l'application Streamlit
if __name__ == '__main__':
    main()