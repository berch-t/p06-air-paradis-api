import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import json
import time
import os
from datetime import datetime

# Configuration de l'URL de l'API
API_URL = st.secrets.get("API_URL", "http://localhost:5000")

# Chemin du fichier de sauvegarde des prédictions
PREDICTIONS_DIR = os.path.join(os.path.expanduser("~"), ".air_paradis")
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
PREDICTIONS_FILE = os.path.join(PREDICTIONS_DIR, "predictions_history.json")

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Air Paradis - Analyse de Sentiment",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonction pour sauvegarder l'historique des prédictions
def save_predictions_history():
    """
    Sauvegarde l'historique des prédictions dans un fichier JSON
    """
    try:
        with open(PREDICTIONS_FILE, 'w', encoding='utf-8') as f:
            # Conversion des timestamps en chaînes de caractères pour JSON
            history_to_save = []
            for pred in st.session_state.predictions_history:
                pred_copy = pred.copy()
                if isinstance(pred_copy.get('timestamp'), datetime):
                    pred_copy['timestamp'] = pred_copy['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                history_to_save.append(pred_copy)
            
            json.dump(history_to_save, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde des prédictions: {str(e)}")
        return False

# Fonction pour charger l'historique des prédictions
def load_predictions_history():
    """
    Charge l'historique des prédictions depuis un fichier JSON
    """
    if not os.path.exists(PREDICTIONS_FILE):
        return []
    
    try:
        with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Erreur lors du chargement des prédictions: {str(e)}")
        return []

# Fonction pour prédire le sentiment
def predict_sentiment(tweet):
    """
    Envoie une requête à l'API pour prédire le sentiment d'un tweet
    """
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"tweet": tweet},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur lors de la prédiction: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Erreur de connexion à l'API: {str(e)}")
        return None

# Fonction pour envoyer un feedback
def send_feedback(tweet, predicted_sentiment, is_correct):
    """
    Envoie un feedback à l'API sur la qualité de la prédiction
    """
    try:
        response = requests.post(
            f"{API_URL}/feedback",
            json={
                "tweet": tweet,
                "predicted_sentiment": predicted_sentiment,
                "is_correct": is_correct
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return True
        else:
            st.error(f"Erreur lors de l'envoi du feedback: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        st.error(f"Erreur de connexion à l'API: {str(e)}")
        return False

# Fonction pour vérifier l'état de l'API
def check_api_health():
    """
    Vérifie si l'API est disponible
    """
    try:
        response = requests.get(f"{API_URL}/health")
        return response.status_code == 200
    except:
        return False

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #003366;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0066cc;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .sentiment-positive {
        color: green;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .sentiment-negative {
        color: red;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .confidence-high {
        color: green;
    }
    .confidence-medium {
        color: orange;
    }
    .confidence-low {
        color: red;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #666;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Entête
st.markdown('<h1 class="main-header">✈️ Air Paradis - Analyse de Sentiment</h1>', unsafe_allow_html=True)

# Vérification de l'état de l'API
api_status = check_api_health()
if not api_status:
    st.error("❌ L'API d'analyse de sentiment n'est pas disponible actuellement. Veuillez réessayer plus tard.")
    st.stop()
else:
    st.success("✅ Connexion à l'API établie avec succès.")

# Barre latérale
st.sidebar.image("air-paradis-logo.png", width=150)
st.sidebar.markdown("## À propos")
st.sidebar.info(
    "Cette application permet d'analyser le sentiment des tweets concernant la compagnie aérienne Air Paradis. "
    "Elle utilise un modèle d'intelligence artificielle pour déterminer si un tweet exprime un sentiment positif ou négatif."
)

st.sidebar.markdown("## Comment ça marche")
st.sidebar.markdown(
    "1. Entrez un tweet dans le champ de texte\n"
    "2. Cliquez sur 'Analyser le sentiment'\n"
    "3. Consultez les résultats de l'analyse\n"
    "4. Donnez votre feedback sur la qualité de la prédiction"
)

# Initialisation de l'historique des prédictions
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = load_predictions_history()

# Statistiques dans la barre latérale
st.sidebar.markdown("## Statistiques")
total_predictions = len(st.session_state.predictions_history)
if total_predictions > 0:
    positive_count = sum(1 for p in st.session_state.predictions_history if p.get('sentiment') == 'Positif')
    negative_count = total_predictions - positive_count
    correct_predictions = sum(1 for p in st.session_state.predictions_history if p.get('feedback') == True)
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    
    st.sidebar.metric("Total de prédictions", total_predictions)
    st.sidebar.metric("Sentiments positifs", f"{positive_count} ({positive_count/total_predictions*100:.1f}%)")
    st.sidebar.metric("Sentiments négatifs", f"{negative_count} ({negative_count/total_predictions*100:.1f}%)")
    st.sidebar.metric("Précision (selon feedback)", f"{accuracy:.1f}%")
else:
    st.sidebar.info("Aucune prédiction n'a encore été effectuée.")

# Zone principale
st.markdown('<h2 class="sub-header">Analyse d\'un nouveau tweet</h2>', unsafe_allow_html=True)

# Entrée du tweet
tweet_input = st.text_area("Entrez un tweet à analyser", height=100)

# Analyse du sentiment
if st.button("Analyser le sentiment", key="btn_analyze"):
    if tweet_input:
        with st.spinner("Analyse en cours..."):
            # Appel à l'API pour la prédiction
            result = predict_sentiment(tweet_input)
            
            if result:
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
                
                # Affichage du résultat
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"**Tweet analysé :** {tweet_input}", unsafe_allow_html=True)
                st.markdown(f"**Tweet prétraité :** {result.get('processed_tweet', '')}", unsafe_allow_html=True)
                st.markdown(f"**Sentiment détecté :** <span class='{sentiment_class}'>{sentiment}</span>", unsafe_allow_html=True)
                st.markdown(f"**Niveau de confiance :** <span class='{confidence_class}'>{confidence:.1f}%</span>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Demande de feedback
                st.markdown("### Êtes-vous d'accord avec cette prédiction ?")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("✅ Oui, c'est correct", key="btn_correct"):
                        feedback_success = send_feedback(tweet_input, sentiment, True)
                        if feedback_success:
                            st.success("Merci pour votre feedback !")
                            # Sauvegarde de la prédiction et du feedback dans l'historique
                            result['feedback'] = True
                            result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.session_state.predictions_history.append(result)
                            # Sauvegarde des prédictions dans un fichier
                            save_predictions_history()
                            # Force refresh the sidebar stats
                            st.rerun()
                
                with col2:
                    if st.button("❌ Non, c'est incorrect", key="btn_incorrect"):
                        feedback_success = send_feedback(tweet_input, sentiment, False)
                        if feedback_success:
                            st.warning("Merci pour votre feedback ! Nous utiliserons cette information pour améliorer notre modèle.")
                            # Sauvegarde de la prédiction et du feedback dans l'historique
                            result['feedback'] = False
                            result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.session_state.predictions_history.append(result)
                            # Sauvegarde des prédictions dans un fichier
                            save_predictions_history()
                            # Force refresh the sidebar stats
                            st.rerun()
    else:
        st.warning("Veuillez entrer un tweet à analyser.")

# Historique des prédictions
if st.session_state.predictions_history:
    st.markdown('<h2 class="sub-header">Historique des prédictions</h2>', unsafe_allow_html=True)
    
    # Création d'un DataFrame pour l'historique
    history_df = pd.DataFrame(st.session_state.predictions_history)
    
    # Affichage de l'historique
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Répartition des sentiments
        sentiment_counts = history_df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        
        fig = px.pie(
            sentiment_counts, 
            values='Count', 
            names='Sentiment', 
            title='Répartition des sentiments',
            color='Sentiment',
            color_discrete_map={'Positif': 'green', 'Négatif': 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Précision du modèle
        feedback_counts = history_df['feedback'].value_counts().reset_index()
        feedback_counts.columns = ['Correct', 'Count']
        
        fig = px.pie(
            feedback_counts, 
            values='Count', 
            names='Correct', 
            title='Précision du modèle (selon le feedback)',
            color='Correct',
            color_discrete_map={True: 'green', False: 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)

# Pied de page
st.markdown('<div class="footer">© 2025 Air Paradis - Analyse de Sentiment | Développé par MIC (Marketing Intelligence Consulting)</div>', 
            unsafe_allow_html=True)