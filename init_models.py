"""
Script pour pré-télécharger le modèle DistilBERT au démarrage
Configuration basée sur le notebook 4_modele_bert.py
"""
import os
import logging
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import numpy as np
import gc  # Pour le garbage collector

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration TensorFlow pour utiliser la mémoire GPU de manière efficace
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
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)

# Paramètres adaptés du notebook 4_modele_bert.py
MAX_SEQ_LENGTH = 64  # Longueur maximale des séquences réduite pour économiser la mémoire
BATCH_SIZE = 8      # Taille de batch réduite pour économiser la mémoire

def download_models():
    """Télécharger et sauvegarder le modèle DistilBERT"""
    try:
        logger.info("Début du téléchargement du modèle DistilBERT...")
        
        # Vérifier et créer le dossier 'models/bert' s'il n'existe pas
        if not os.path.exists('models'):
            os.makedirs('models')
        if not os.path.exists('models/bert'):
            os.makedirs('models/bert')
            
        # Téléchargement du modèle DistilBERT si nécessaire
        if not (os.path.exists('models/bert/tokenizer_bert') and os.path.exists('models/bert/best_model_bert')):
            logger.info("Téléchargement du tokenizer DistilBERT...")
            
            # DistilBERT comme utilisé dans le notebook
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', cache_dir='./models/bert/cache')
            tokenizer.save_pretrained('models/bert/tokenizer_bert')
            
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
            
            # Créer un petit lot de données d'exemple pour initialiser les poids
            dummy_input_ids = tf.random.uniform(shape=[1, MAX_SEQ_LENGTH], dtype=tf.int32, minval=0, maxval=tokenizer.vocab_size)
            dummy_attention_mask = tf.ones(shape=[1, MAX_SEQ_LENGTH], dtype=tf.int32)
            
            # Faire une passe avant pour initialiser les poids
            _ = model({
                'input_ids': dummy_input_ids,
                'attention_mask': dummy_attention_mask
            }, training=False)
            
            # Sauvegarde du modèle
            model.save_pretrained('models/bert/best_model_bert')
            
            # Libérer la mémoire
            del model
            gc.collect()
            
            logger.info("Modèle DistilBERT téléchargé et sauvegardé avec succès!")
        else:
            logger.info("Le modèle DistilBERT est déjà présent.")
            
        # Sauvegarder les paramètres importants
        config = {
            'max_sequence_length': MAX_SEQ_LENGTH,
            'batch_size': BATCH_SIZE,
            'model_type': 'distilbert-base-uncased'
        }
        
        # Sauvegarde de la configuration
        import json
        with open('models/bert/config.json', 'w') as f:
            json.dump(config, f)
        
        logger.info("Configuration sauvegardée avec succès!")
            
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement des modèles: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        raise
    finally:
        # Forcer le garbage collector
        gc.collect()

if __name__ == "__main__":
    download_models()