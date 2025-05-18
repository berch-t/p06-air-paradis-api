"""
Script pour pré-télécharger les modèles au démarrage
Configuration basée sur le notebook 3_modele_avance.py

Ce script est exécuté pendant le déploiement pour éviter les temps d'attente 
lors de la première requête.
"""
import os
import logging
import tensorflow as tf
import numpy as np
import pickle
import gc  # Pour le garbage collector

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration TensorFlow pour utiliser la mémoire de manière efficace
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Pour limiter l'utilisation de la mémoire GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"GPUs disponibles: {len(gpus)}")
    except RuntimeError as e:
        logger.error(e)
else:
    logger.info("Aucun GPU détecté, utilisation du CPU")
    # Limiter l'utilisation du CPU pour éviter les crashs
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)

# Paramètres adaptés du notebook 3_modele_avance.py 
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 300

def create_custom_model():
    """
    Création d'un modèle CNN-LSTM optimisé basé sur le modèle dans 3_modele_avance.py
    """
    try:
        # Modèle CNN-LSTM
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                input_dim=MAX_NUM_WORDS + 1,
                output_dim=EMBEDDING_DIM,
                input_length=MAX_SEQUENCE_LENGTH,
                trainable=True
            ),
            tf.keras.layers.SpatialDropout1D(0.3),
            # Partie CNN
            tf.keras.layers.Conv1D(128, 5, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(5),
            tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(3),
            # Partie LSTM
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=0.3, recurrent_dropout=0.3)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compilation du modèle
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    except Exception as e:
        logger.error(f"Erreur lors de la création du modèle: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        raise

def load_tokenizer():
    """Charge le tokenizer préentraîné"""
    try:
        if os.path.exists('models/tokenizer.pickle'):
            with open('models/tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
            logger.info("Tokenizer chargé avec succès")
            return tokenizer
        else:
            logger.warning("Fichier du tokenizer non trouvé")
            return None
    except Exception as e:
        logger.error(f"Erreur lors du chargement du tokenizer: {str(e)}")
        return None

def initialize_models():
    """Initialise les modèles pour le déploiement"""
    try:
        logger.info("Initialisation des modèles...")
        
        # Vérifier et créer le dossier 'models' s'il n'existe pas
        if not os.path.exists('models'):
            os.makedirs('models')
            
        # Définir le nombre maximum de mots (même valeur que dans le notebook)
        global MAX_NUM_WORDS
        MAX_NUM_WORDS = 30000
        
        # Charger le tokenizer
        tokenizer = load_tokenizer()
        
        # Vérifier si le modèle pré-entraîné existe
        if os.path.exists('models/best_model'):
            logger.info("Modèle pré-entraîné trouvé, chargement...")
            try:
                model = tf.keras.models.load_model('models/best_model')
                logger.info("Modèle chargé avec succès")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
                logger.info("Création d'un nouveau modèle...")
                model = create_custom_model()
        else:
            logger.info("Aucun modèle pré-entraîné trouvé, création d'un nouveau modèle...")
            model = create_custom_model()
            
            # Sauvegarder l'architecture du modèle (sans les poids)
            try:
                model_json = model.to_json()
                with open('models/model_architecture.json', 'w') as json_file:
                    json_file.write(model_json)
                logger.info("Architecture du modèle sauvegardée")
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde de l'architecture: {str(e)}")
        
        # Sauvegarde de la configuration
        config = {
            'max_sequence_length': MAX_SEQUENCE_LENGTH,
            'embedding_dim': EMBEDDING_DIM,
            'max_num_words': MAX_NUM_WORDS,
            'model_type': 'CNN-LSTM'
        }
        
        # Sauvegarde de la configuration
        import json
        with open('models/config.json', 'w') as f:
            json.dump(config, f)
        
        logger.info("Initialisation des modèles terminée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation des modèles: {str(e)}")
        logger.error("Traceback:", exc_info=True)
    finally:
        # Forcer le garbage collector
        gc.collect()

if __name__ == "__main__":
    initialize_models()