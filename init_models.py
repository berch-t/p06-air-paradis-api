"""
Script pour pré-télécharger les modèles BERT au démarrage
Configuration exacte basée sur le notebook 4_modele_bert

Ce script peut être exécuté pendant le déploiement pour éviter les temps d'attente lors de la première requête
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
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)

# Paramètres adaptés du notebook 4_modele_bert.py mis à jour
MAX_SEQ_LENGTH = 64  # Longueur maximale des séquences réduite pour économiser la mémoire
BATCH_SIZE = 8      # Taille de batch réduite pour économiser la mémoire

def create_custom_bert_model():
    """
    Création d'un modèle DistilBERT avec une stratégie d'optimisation personnalisée
    Basé exactement sur le modèle optimisé dans le notebook 4_modele_bert.py mis à jour
    """
    # Utilisation de DistilBERT comme dans le notebook
    try:
        # Configuration d'un learning rate adaptatif comme dans le notebook
        class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, d_model, warmup_steps=4000):
                super(CustomSchedule, self).__init__()
                self.d_model = d_model
                self.d_model = tf.cast(self.d_model, tf.float32)
                self.warmup_steps = warmup_steps
                
            def __call__(self, step):
                step = tf.cast(step, tf.float32)
                arg1 = tf.math.rsqrt(step)
                arg2 = step * (self.warmup_steps ** -1.5)
                return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        
        # Création du modèle DistilBERT
        model = TFDistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', 
            num_labels=2,
            cache_dir='./distilbert_cache'
        )
        
        # Pour la production, on utilisera un Learning Rate fixe simple comme suggéré pour les small samples dans le notebook
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
        
        # Compilation du modèle
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        return model
    except Exception as e:
        logger.error(f"Erreur lors de la création du modèle DistilBERT: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        raise

def download_models():
    """Télécharger et sauvegarder les modèles DistilBERT optimisés selon le notebook"""
    try:
        logger.info("Début du téléchargement des modèles DistilBERT optimisés...")
        
        # Vérifier et créer le dossier 'models' s'il n'existe pas
        if not os.path.exists('models'):
            os.makedirs('models')
        if not os.path.exists('models/bert'):
            os.makedirs('models/bert')
            
        # Téléchargement du modèle DistilBERT si nécessaire
        if not (os.path.exists('models/bert/tokenizer') and os.path.exists('models/bert/model')):
            logger.info("Téléchargement du tokenizer DistilBERT...")
            
            # DistilBERT comme utilisé dans le notebook
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', cache_dir='./distilbert_cache')
            tokenizer.save_pretrained('models/bert/tokenizer')
            
            logger.info("Configuration du modèle DistilBERT optimisé...")
            model = create_custom_bert_model()
            
            # Créer un petit lot de données d'exemple pour initialiser les poids
            dummy_input_ids = tf.random.uniform(shape=[1, MAX_SEQ_LENGTH], dtype=tf.int32, minval=0, maxval=tokenizer.vocab_size)
            dummy_attention_mask = tf.ones(shape=[1, MAX_SEQ_LENGTH], dtype=tf.int32)
            
            # Faire une passe avant pour initialiser les poids
            _ = model({
                'input_ids': dummy_input_ids,
                'attention_mask': dummy_attention_mask
            }, training=False)
            
            # Sauvegarde du modèle
            model.save_pretrained('models/bert/model')
            
            # Libérer la mémoire
            del model
            gc.collect()
            
            logger.info("Modèle DistilBERT optimisé téléchargé et sauvegardé avec succès!")
        else:
            logger.info("Le modèle DistilBERT est déjà présent.")
            
        # Sauvegarder les paramètres importants
        config = {
            'max_seq_length': MAX_SEQ_LENGTH,
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