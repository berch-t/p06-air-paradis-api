#!/bin/bash
set -e

# Définir des variables d'environnement
export PYTHONUNBUFFERED=1
export PORT=${PORT:-8000}
export PYTHONPATH=$(pwd)
export APPINSIGHTS_INSTRUMENTATION_KEY=${APPINSIGHTS_INSTRUMENTATION_KEY:-"9af56b4d-4ad5-4643-ba29-41d154893ad4"}
export FLASK_APP=application.py

# Créer les répertoires nécessaires
mkdir -p logs
mkdir -p models/bert

# Afficher les informations sur l'environnement
echo "== Environnement d'exécution ==" > logs/startup.log
echo "Python: $(python --version)" >> logs/startup.log
echo "Répertoire de travail: $(pwd)" >> logs/startup.log
echo "Contenu du répertoire:" >> logs/startup.log
ls -la >> logs/startup.log

# Installer les dépendances si nécessaire
if [ ! -f ".dependencies_installed" ]; then
    echo "Installation des packages requis..." >> logs/startup.log
    python -m pip install -r requirements.txt || true
    touch .dependencies_installed
fi

# Initialiser le modèle BERT
echo "== Initialisation du modèle BERT ==" >> logs/startup.log
python init_models.py >> logs/startup.log 2>&1 || true
echo "Initialisation du modèle terminée" >> logs/startup.log

# Vérifier si le modèle a été correctement initialisé
if [ -d "models/bert/best_model_bert" ] && [ -d "models/bert/tokenizer_bert" ]; then
    echo "Modèle BERT correctement initialisé" >> logs/startup.log
    export SIMULATION_MODE=false
else
    echo "Modèle BERT non trouvé ou incomplet, démarrage en mode simulation" >> logs/startup.log
    export SIMULATION_MODE=true
fi

# Démarrer le serveur API avec le PORT correct
echo "== Démarrage du serveur API sur le port $PORT ==" >> logs/startup.log
exec gunicorn --bind=0.0.0.0:$PORT --timeout 600 --access-logfile logs/access.log --error-logfile logs/error.log application:app