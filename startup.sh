#!/bin/bash
set -e

# Définir des variables d'environnement
export PYTHONUNBUFFERED=1
export PORT=${PORT:-8000}
export APPINSIGHTS_INSTRUMENTATION_KEY=${APPINSIGHTS_INSTRUMENTATION_KEY:-"9af56b4d-4ad5-4643-ba29-41d154893ad4"}

# Créer les répertoires nécessaires
mkdir -p logs
mkdir -p models/bert

# Afficher les informations sur l'environnement
echo "== Environnement d'exécution =="
echo "Python: $(python --version)"
echo "Répertoire de travail: $(pwd)"
echo "Contenu du répertoire:"
ls -la

# S'assurer que pip est disponible et mis à jour
echo "== Installation des dépendances =="
which python
python -m pip --version || (echo "Pip n'est pas disponible, tentative d'installation..." && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py)
python -m pip install --upgrade pip

# Installer les dépendances
echo "Installation des packages requis..."
python -m pip install -r requirements.txt

# Vérification des répertoires de modèles
echo "== Vérification des modèles =="
find . -type d | grep models || echo "Aucun répertoire models trouvé"
if [ -d "models/bert" ]; then
    echo "Contenu du répertoire models/bert:"
    ls -la models/bert
else
    echo "Le répertoire models/bert n'existe pas encore"
fi

if [ -d "preloaded_models" ]; then
    echo "Contenu du répertoire preloaded_models:"
    ls -la preloaded_models
fi

# Initialiser le modèle BERT
echo "== Initialisation du modèle BERT =="
python init_models.py > logs/model_init.log 2>&1
echo "Initialisation du modèle terminée"

# Vérifier si le modèle a été correctement initialisé
if [ -d "models/bert/best_model_bert" ] && [ -d "models/bert/tokenizer_bert" ]; then
    echo "Modèle BERT correctement initialisé"
    # Démarrer en mode production
    export SIMULATION_MODE=false
else
    echo "Modèle BERT non trouvé ou incomplet, démarrage en mode simulation"
    export SIMULATION_MODE=true
fi

# Démarrer le serveur API
echo "== Démarrage du serveur API =="
gunicorn --bind=0.0.0.0:$PORT --timeout 600 --workers 1 --log-level debug application:app > logs/api.log 2>&1 &
API_PID=$!
echo "Serveur API démarré avec PID: $API_PID"

# Attendre que l'API soit prête
echo "Attente de l'initialisation de l'API..."
sleep 10

# Vérifier si l'API est active
if ps -p $API_PID > /dev/null; then
    echo "API démarrée avec succès!"
    # Garder le processus actif
    tail -f logs/api.log
else
    echo "ERREUR: Le processus API s'est arrêté prématurément."
    echo "Contenu des logs d'erreur:"
    cat logs/api.log
    exit 1
fi