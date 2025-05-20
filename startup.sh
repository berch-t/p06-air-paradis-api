#!/bin/bash

# Définir des variables d'environnement
export PYTHONUNBUFFERED=1
export PORT=${PORT:-8000}
export APPINSIGHTS_INSTRUMENTATION_KEY="9af56b4d-4ad5-4643-ba29-41d154893ad4"

# Créer les répertoires nécessaires
mkdir -p logs
mkdir -p models/bert

# Afficher les informations sur l'environnement
echo "Environnement Python:"
which python
python --version
echo "Emplacement de pip:"
which pip

# Installer les dépendances avec le chemin complet de pip
echo "Installation des dépendances requises..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Initialiser le modèle BERT
echo "Initialisation du modèle BERT..."
python init_models.py

# Démarrer le serveur API
echo "Démarrage du serveur API..."
gunicorn --bind=0.0.0.0:$PORT --timeout 600 --workers 2 application:app > logs/api.log 2>&1 &
API_PID=$!
echo "Serveur API démarré avec PID: $API_PID"

# Attendre que l'API soit prête
echo "Attente de l'initialisation de l'API..."
sleep 10

# Vérifier si l'API est active
if ! ps -p $API_PID > /dev/null; then
    echo "Avertissement: Le processus du serveur API s'est arrêté. Vérification des logs:"
    cat logs/api.log
    echo "Tentative de redémarrage de l'API avec plus de debug..."
    gunicorn --bind=0.0.0.0:$PORT --timeout 600 --workers 1 --log-level debug application:app
fi

# Garder le processus actif
tail -f logs/api.log