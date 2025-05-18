"""
Streamlit application for sentiment analysis of tweets using a trained BiLSTM model
"""
import os
import sys
import re
import logging
import pickle
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, Union, TypedDict

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import requests
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("streamlit_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Application Insights configuration
APPINSIGHTS_INSTRUMENTATION_KEY = os.getenv('APPINSIGHTS_INSTRUMENTATION_KEY', '')
APPINSIGHTS_ENDPOINT = "https://dc.services.visualstudio.com/v2/track"

# Type definitions
class TelemetryProperties(TypedDict, total=False):
    tweet: str
    sentiment: str
    confidence: float
    model: str
    predicted_sentiment: str
    is_incorrect: str

class PredictionResult(TypedDict):
    tweet: str
    processed_tweet: str
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]

# Telemetry functions
def send_appinsights_telemetry(event_name: str, properties: Optional[TelemetryProperties] = None) -> None:
    """
    Send telemetry data to Azure Application Insights
    
    Args:
        event_name: Name of the telemetry event
        properties: Dictionary of properties to include with the event
    """
    if not APPINSIGHTS_INSTRUMENTATION_KEY:
        logger.warning("Azure Application Insights instrumentation key not set. Telemetry will not be sent.")
        return
    
    try:
        telemetry_data = {
            "name": "Microsoft.ApplicationInsights.Event",
            "time": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "iKey": APPINSIGHTS_INSTRUMENTATION_KEY,
            "tags": {
                "ai.cloud.roleInstance": "streamlit-sentiment-analysis"
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
        
        response = requests.post(
            APPINSIGHTS_ENDPOINT,
            json=telemetry_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            logger.error(f"Error sending telemetry: {response.status_code} - {response.text}")
        else:
            logger.info(f"Telemetry '{event_name}' sent successfully")
            
    except Exception as e:
        logger.error(f"Error sending telemetry: {str(e)}")

# Model functions
def preprocess_tweet(tweet: str) -> str:
    """
    Preprocess a tweet for sentiment analysis
    
    Args:
        tweet: Raw tweet text
        
    Returns:
        Preprocessed tweet text
    """
    # Convert to lowercase
    tweet = tweet.lower()
    
    # Remove mentions
    tweet = re.sub(r'@\w+', '', tweet)
    
    # Remove URLs
    tweet = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', tweet)
    
    # Remove hashtags (keep the word without #)
    tweet = re.sub(r'#(\w+)', r'\1', tweet)
    
    # Remove special characters and punctuation
    tweet = re.sub(r'[^\w\s]', '', tweet)
    
    # Remove numbers
    tweet = re.sub(r'\d+', '', tweet)
    
    return tweet.strip()

def load_model() -> Tuple[Optional[tf.keras.Model], Optional[Any], Optional[Dict[str, Any]]]:
    """
    Load the BiLSTM model with Word2Vec for sentiment analysis
    
    Returns:
        Tuple of (model, tokenizer, config) or (None, None, None) if loading fails
    """
    try:
        # Paths to model files
        model_path = 'models/best_advanced_model_BiLSTM_Word2Vec.h5'
        tokenizer_path = 'models/tokenizer.pickle'
        config_path = 'models/model_config.pickle'
        
        # Check if files exist
        if not os.path.exists(model_path):
            logger.error(f"BiLSTM model not found. Check that the file exists: {model_path}")
            return None, None, None
        
        if not os.path.exists(tokenizer_path):
            logger.error(f"Tokenizer not found. Check that the file exists: {tokenizer_path}")
            return None, None, None
            
        if not os.path.exists(config_path):
            logger.error(f"Model config not found. Check that the file exists: {config_path}")
            return None, None, None
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Load tokenizer
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
            
        # Load config
        with open(config_path, 'rb') as handle:
            config = pickle.load(handle)
        
        logger.info("BiLSTM model, tokenizer, and config loaded successfully")
        return model, tokenizer, config
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None, None

def predict_sentiment(tweet: str, max_length: int = 50) -> PredictionResult:
    """
    Predict the sentiment of a tweet using the BiLSTM model
    
    Args:
        tweet: Tweet text to analyze
        max_length: Maximum sequence length for padding
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Preprocess the tweet
        processed_tweet = preprocess_tweet(tweet)
        
        # Load model and tokenizer
        model, tokenizer, config = load_model()
        
        # If model or tokenizer is not available, use demo mode
        if model is None or tokenizer is None or config is None:
            return predict_demo(tweet)
        
        # Get max sequence length from config if available
        max_length = config.get('max_sequence_length', max_length)
        
        # Tokenize
        sequences = tokenizer.texts_to_sequences([processed_tweet])
        
        # Padding
        padded_sequences = pad_sequences(
            sequences,
            maxlen=max_length,
            padding='post'
        )
        
        # Prediction
        prediction = model.predict(padded_sequences)
        probability = prediction[0][0]
        
        # Log information
        logger.info(f"Raw tweet: {tweet}")
        logger.info(f"Preprocessed tweet: {processed_tweet}")
        logger.info(f"Positive probability: {probability:.4f}")
        
        # Decision with standard threshold
        threshold = 0.5
        sentiment = "Positif" if probability > threshold else "Négatif"
        confidence = float(probability if sentiment == "Positif" else 1 - probability)
        
        # Create response
        result: PredictionResult = {
            'tweet': tweet,
            'processed_tweet': processed_tweet,
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                'positive': float(probability),
                'negative': float(1 - probability)
            }
        }
        
        # Send telemetry
        send_appinsights_telemetry("ModelPrediction", {
            "tweet": tweet,
            "sentiment": sentiment,
            "confidence": confidence,
            "model": "BiLSTM_Word2Vec"
        })
        
        return result
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.info("Using demo mode due to an error")
        return predict_demo(tweet)

def predict_demo(tweet: str) -> PredictionResult:
    """
    Demo prediction without loading a model
    
    Args:
        tweet: Tweet text to analyze
        
    Returns:
        Dictionary with demo prediction results
    """
    # Calculate a random sentiment for demonstration
    positive_score = np.random.random()
    sentiment = "Positif" if positive_score > 0.5 else "Négatif"
    confidence = positive_score if sentiment == "Positif" else 1 - positive_score
    
    processed_tweet = preprocess_tweet(tweet)
    
    # Send telemetry for demo prediction
    send_appinsights_telemetry("DemoPrediction", {
        "tweet": tweet,
        "sentiment": sentiment,
        "confidence": confidence,
        "model": "Demo"
    })
    
    return {
        'tweet': tweet,
        'processed_tweet': processed_tweet,
        'sentiment': sentiment,
        'confidence': confidence,
        'probabilities': {
            'positive': float(positive_score),
            'negative': float(1 - positive_score)
        }
    }

def generate_sample_data(size: int = 20) -> pd.DataFrame:
    """
    Generate sample data for visualizations
    
    Args:
        size: Number of sample data points to generate
        
    Returns:
        DataFrame with sample data
    """
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
        
        # Direct sentiment without calling the model
        sentiment = "Positif" if idx < 3 else "Négatif"
        confidence = np.random.uniform(0.7, 0.95)
        
        # Random date in the last 30 days
        days_ago = np.random.randint(1, 31)
        date = (datetime.now() - pd.Timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        data.append({
            'tweet': tweet,
            'sentiment': sentiment,
            'confidence': confidence,
            'date': date
        })
    
    return pd.DataFrame(data)

# UI Components
def render_header() -> None:
    """Render the application header"""
    st.markdown('<h1 class="header">✈️ Air Paradis - Sentiment Analysis</h1>', unsafe_allow_html=True)

def render_sidebar() -> None:
    """Render the sidebar with configuration options and model stats"""
    with st.sidebar:
        st.title("Configuration")
        
        # Get global reference before using it
        global APPINSIGHTS_INSTRUMENTATION_KEY
        
        # Azure Application Insights instrumentation key
        appinsights_key = st.text_input(
            "Azure Application Insights Instrumentation Key",
            value=APPINSIGHTS_INSTRUMENTATION_KEY,
            type="password"
        )
        
        if appinsights_key:
            os.environ['APPINSIGHTS_INSTRUMENTATION_KEY'] = appinsights_key
            # Update the global variable
            APPINSIGHTS_INSTRUMENTATION_KEY = appinsights_key
            st.success("Instrumentation key updated!")
        
        st.markdown("---")
        
        # Model statistics
        st.subheader("Model Statistics")
        if st.session_state.model_accuracy['total'] > 0:
            accuracy = st.session_state.model_accuracy['correct'] / st.session_state.model_accuracy['total'] * 100
            st.metric("Model Accuracy", f"{accuracy:.1f}%")
            st.metric("Validated Predictions", st.session_state.model_accuracy['correct'])
            st.metric("Total Predictions", st.session_state.model_accuracy['total'])
        else:
            st.info("No data available")

def render_analysis_section() -> None:
    """Render the tweet analysis section"""
    st.subheader("Analyze a new tweet")
    tweet_input = st.text_area("Enter a tweet to analyze", height=100)

    if st.button("Analyze Sentiment", type="primary"):
        if tweet_input:
            with st.spinner("Analysis in progress..."):
                try:
                    # Predict sentiment using the model or demo mode if model not available
                    result = predict_sentiment(tweet_input)
                    
                    # Store in history
                    result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    result['feedback'] = None
                    st.session_state.predictions_history.append(result)
                    
                    # Display results
                    sentiment = result['sentiment']
                    confidence = result['confidence'] * 100
                    
                    st.write(f"**Tweet:** {tweet_input}")
                    st.write(f"**Preprocessed Tweet:** {result['processed_tweet']}")
                    
                    if sentiment == "Positif":
                        st.markdown(f"**Sentiment:** <span class='positive'>{sentiment}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**Sentiment:** <span class='negative'>{sentiment}</span>", unsafe_allow_html=True)
                    
                    st.write(f"**Confidence:** {confidence:.1f}%")
                    
                    # Probability visualization
                    fig = go.Figure(go.Bar(
                        x=[result['probabilities']['positive'], result['probabilities']['negative']],
                        y=['Positive', 'Negative'],
                        orientation='h',
                        marker=dict(color=['#059669', '#DC2626'])
                    ))
                    
                    fig.update_layout(
                        title='Probability by Class',
                        xaxis_title='Probability',
                        yaxis_title='Sentiment',
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feedback
                    st.subheader("Do you agree with this prediction?")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("✅ Yes, it's correct"):
                            prediction_index = len(st.session_state.predictions_history) - 1
                            st.session_state.predictions_history[prediction_index]['feedback'] = True
                            
                            # Update model statistics
                            st.session_state.model_accuracy['correct'] += 1
                            st.session_state.model_accuracy['total'] += 1
                            
                            # Send telemetry for positive feedback
                            send_appinsights_telemetry("PositiveFeedback", {
                                "tweet": tweet_input,
                                "sentiment": sentiment,
                                "confidence": result['confidence']
                            })
                            
                            st.success("Thank you for your feedback!")
                            st.experimental_rerun()
                    
                    with col2:
                        if st.button("❌ No, it's incorrect"):
                            prediction_index = len(st.session_state.predictions_history) - 1
                            st.session_state.predictions_history[prediction_index]['feedback'] = False
                            
                            # Update model statistics
                            st.session_state.model_accuracy['total'] += 1
                            
                            # Send telemetry for negative feedback
                            send_appinsights_telemetry("NegativeFeedback", {
                                "tweet": tweet_input,
                                "sentiment": sentiment,
                                "confidence": result['confidence']
                            })
                            
                            st.warning("Thank you for your feedback! We'll use this information to improve our model.")
                            st.experimental_rerun()
                
                except Exception as e:
                    logger.error(f"Error during analysis: {str(e)}")
                    st.error(f"Error during analysis: {str(e)}")
        else:
            st.warning("Please enter a tweet to analyze.")

def render_history_and_visualizations() -> None:
    """Render the prediction history and visualizations"""
    if st.session_state.predictions_history:
        st.subheader("Prediction History")
        
        # Create DataFrame
        history_df = pd.DataFrame(st.session_state.predictions_history)
        
        # Display history
        with st.expander("View History"):
            # Prepare data for display
            display_df = history_df.copy()
            
            if 'probabilities' in display_df.columns:
                display_df = display_df.drop(columns=['probabilities'])
            
            if 'confidence' in display_df.columns:
                display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.1f}%" if isinstance(x, (float, int)) else x)
            
            # Ensure feedback column exists
            if 'feedback' not in display_df.columns:
                display_df['feedback'] = None
            
            # Display
            cols_to_display = ['tweet', 'sentiment', 'confidence', 'feedback', 'timestamp']
            cols_to_display = [col for col in cols_to_display if col in display_df.columns]
            
            rename_dict = {
                'tweet': 'Tweet',
                'sentiment': 'Sentiment',
                'confidence': 'Confidence',
                'feedback': 'Correct Feedback',
                'timestamp': 'Timestamp'
            }
            
            st.dataframe(
                display_df[cols_to_display].rename(columns=rename_dict),
                use_container_width=True
            )
        
        # Basic visualizations
        st.subheader("Visualizations")
        
        # Option to generate sample data
        if len(st.session_state.predictions_history) < 5:
            st.info("Not enough data for meaningful visualizations.")
            if st.button("Generate Sample Data"):
                sample_df = generate_sample_data(20)
                
                for _, row in sample_df.iterrows():
                    st.session_state.predictions_history.append({
                        'tweet': row['tweet'],
                        'sentiment': row['sentiment'],
                        'confidence': row['confidence'],
                        'feedback': True,
                        'timestamp': row['date']
                    })
                
                # Update model statistics
                st.session_state.model_accuracy['correct'] += len(sample_df)
                st.session_state.model_accuracy['total'] += len(sample_df)
                
                st.success("Sample data generated successfully!")
                st.experimental_rerun()
        else:
            # Simple visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment distribution
                sentiment_counts = history_df['sentiment'].value_counts()
                fig = px.pie(
                    names=sentiment_counts.index,
                    values=sentiment_counts.values,
                    title="Sentiment Distribution",
                    color=sentiment_counts.index,
                    color_discrete_map={'Positif': '#059669', 'Négatif': '#DC2626'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence distribution
                if 'confidence' in history_df.columns:
                    fig = px.histogram(
                        history_df,
                        x='confidence',
                        color='sentiment',
                        title="Confidence Score Distribution",
                        color_discrete_map={'Positif': '#059669', 'Négatif': '#DC2626'},
                        nbins=10
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Feedback and model accuracy
            if 'feedback' in history_df.columns:
                feedback_counts = history_df['feedback'].value_counts()
                
                if True in feedback_counts and False in feedback_counts:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        feedback_df = pd.DataFrame({
                            'Feedback': ['Correct', 'Incorrect'],
                            'Count': [feedback_counts.get(True, 0), feedback_counts.get(False, 0)]
                        })
                        
                        fig = px.bar(
                            feedback_df,
                            x='Feedback',
                            y='Count',
                            title="User Feedback",
                            color='Feedback',
                            color_discrete_map={'Correct': '#059669', 'Incorrect': '#DC2626'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        total = feedback_counts.get(True, 0) + feedback_counts.get(False, 0)
                        accuracy = feedback_counts.get(True, 0) / total * 100 if total > 0 else 0
                        
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = accuracy,
                            title = {'text': "Model Accuracy (%)"},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#1E3A8A"},
                                'steps': [
                                    {'range': [0, 50], 'color': "#DC2626"},
                                    {'range': [50, 80], 'color': "#FBBF24"},
                                    {'range': [80, 100], 'color': "#059669"}
                                ]
                            }
                        ))
                        
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)

def render_appinsights_info() -> None:
    """Render information about Azure Application Insights configuration"""
    with st.expander("How to configure Azure Application Insights"):
        st.markdown("""
        ### Azure Application Insights Configuration
        
        To track model performance in production, this application integrates with Azure Application Insights.
        
        #### Steps to configure Azure Application Insights:
        
        1. Create an Application Insights service in the Azure portal
        2. Retrieve the instrumentation key
        3. Enter it in the text field in the sidebar
        
        #### Telemetry sent:
        
        - **ModelPrediction**: When a prediction is made with the model
        - **DemoPrediction**: When a prediction is made in demo mode
        - **PositiveFeedback**: When a user validates a prediction
        - **NegativeFeedback**: When a user indicates a prediction is incorrect
        
        These telemetry events can be used to set up alerts in Azure Application Insights, for example to be notified if the negative feedback rate exceeds a certain threshold.
        """)

def main() -> None:
    """Main Streamlit application for tweet sentiment analysis"""
    # Page configuration
    st.set_page_config(
        page_title="Air Paradis - Sentiment Analysis",
        page_icon="✈️",
        layout="wide"
    )

    # Initialize session state variables
    if 'predictions_history' not in st.session_state:
        st.session_state.predictions_history = []
    if 'model_accuracy' not in st.session_state:
        st.session_state.model_accuracy = {'correct': 0, 'total': 0}

    # Minimal CSS
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
    .appinsights-key {
        margin-top: 1rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Render UI components
    render_header()
    render_sidebar()
    render_analysis_section()
    render_history_and_visualizations()
    render_appinsights_info()

if __name__ == '__main__':
    main() 