"""
Script pour pré-télécharger le modèle DistilBERT au démarrage
Configuration basée sur le notebook 4_modele_bert.py
"""
import os
import logging
import json
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import numpy as np
import gc  # Pour le garbage collector
import shutil  # Pour la copie de fichiers

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("init_models.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"GPUs disponibles: {len(gpus)}")
    except RuntimeError as e:
        logger.error(e)
else:
    logger.info("Aucun GPU détecté, utilisation du CPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)

# Paramètres du modèle
MAX_SEQ_LENGTH = 64
BATCH_SIZE = 8

def check_model_exists():
    """Vérifie si le modèle et le tokenizer existent déjà dans le dossier models/bert"""
    tokenizer_path = os.path.join('models', 'bert', 'tokenizer_bert')
    model_path = os.path.join('models', 'bert', 'best_model_bert')
    
    # Vérifier l'existence des dossiers et fichiers essentiels
    tokenizer_exists = os.path.exists(tokenizer_path) and len(os.listdir(tokenizer_path)) > 0
    model_exists = os.path.exists(model_path) and len(os.listdir(model_path)) > 0
    
    return tokenizer_exists and model_exists

def create_config_file():
    """Crée le fichier de configuration du modèle"""
    config = {
        'max_sequence_length': MAX_SEQ_LENGTH,
        'batch_size': BATCH_SIZE,
        'model_type': 'distilbert-base-uncased'
    }
    
    os.makedirs('models/bert', exist_ok=True)
    with open('models/bert/config.json', 'w') as f:
        json.dump(config, f)
    
    logger.info("Fichier de configuration créé")
    return config

def download_model_from_huggingface():
    """Télécharge le modèle depuis Hugging Face (seulement si nécessaire)"""
    try:
        logger.info("Téléchargement du tokenizer DistilBERT...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', cache_dir='./models/bert/cache')
        tokenizer.save_pretrained('models/bert/tokenizer_bert')
        logger.info("Tokenizer sauvegardé avec succès")
        
        logger.info("Téléchargement du modèle DistilBERT...")
        model = TFDistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', 
            num_labels=2,
            cache_dir='./models/bert/cache'
        )
        
        # Compilation du modèle
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        # Initialisation avec des données factices
        dummy_input_ids = tf.random.uniform(shape=[1, MAX_SEQ_LENGTH], dtype=tf.int32, minval=0, maxval=tokenizer.vocab_size)
        dummy_attention_mask = tf.ones(shape=[1, MAX_SEQ_LENGTH], dtype=tf.int32)
        
        _ = model({
            'input_ids': dummy_input_ids,
            'attention_mask': dummy_attention_mask
        }, training=False)
        
        # Sauvegarde du modèle
        model.save_pretrained('models/bert/best_model_bert')
        logger.info("Modèle sauvegardé avec succès")
        
        # Libération de la mémoire
        del model
        gc.collect()
        
        return True
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement du modèle: {str(e)}")
        return False

def initialize_model():
    """Initialise le modèle DistilBERT"""
    try:
        logger.info("Initialisation du modèle DistilBERT...")
        
        # Création des répertoires si nécessaire
        os.makedirs('models/bert', exist_ok=True)
        
        # Vérification si le modèle existe déjà
        if check_model_exists():
            logger.info("Le modèle DistilBERT est déjà présent dans le dépôt")
            create_config_file()
            return True
        
        # Recherche du modèle dans le dossier de modèles préchargés (pour l'ajout au dépôt)
        preloaded_model_path = os.path.join('preloaded_models', 'bert')
        if os.path.exists(preloaded_model_path):
            logger.info("Utilisation du modèle préchargé dans preloaded_models/bert")
            
            # Copie du tokenizer s'il existe
            preloaded_tokenizer = os.path.join(preloaded_model_path, 'tokenizer_bert')
            if os.path.exists(preloaded_tokenizer):
                logger.info("Copie du tokenizer préchargé...")
                shutil.copytree(preloaded_tokenizer, 'models/bert/tokenizer_bert', dirs_exist_ok=True)
            
            # Copie du modèle s'il existe
            preloaded_model = os.path.join(preloaded_model_path, 'best_model_bert')
            if os.path.exists(preloaded_model):
                logger.info("Copie du modèle préchargé...")
                shutil.copytree(preloaded_model, 'models/bert/best_model_bert', dirs_exist_ok=True)
            
            create_config_file()
            return True
        
        # Si le modèle n'est pas trouvé localement, essayer de le télécharger
        logger.info("Modèle non trouvé localement, tentative de téléchargement...")
        if download_model_from_huggingface():
            return True
        
        # Si tout échoue, créer au moins le fichier de configuration
        logger.warning("Impossible de télécharger le modèle, création d'une configuration minimale")
        create_config_file()
        return False
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du modèle: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        create_config_file()
        return False

if __name__ == "__main__":
    initialize_model()