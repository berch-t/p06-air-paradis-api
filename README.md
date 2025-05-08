# Air Paradis - API d'Analyse de Sentiment

Cette application fournit une API REST et une interface utilisateur Streamlit pour analyser le sentiment de tweets concernant la compagnie aérienne Air Paradis. Elle utilise un modèle BERT avancé pour déterminer si un tweet exprime un sentiment positif ou négatif.

## Fonctionnalités

- **API REST** pour l'analyse de sentiment (/predict, /health, /feedback)
- **Interface utilisateur Streamlit** moderne et interactive
- Analyse de sentiment basée sur **BERT**, l'état de l'art en NLP
- Visualisations dynamiques des résultats et statistiques
- Suivi des performances via **Azure Application Insights**
- Feedback utilisateur pour améliorer le modèle

## Installation et lancement local

### Prérequis

- Python 3.9+
- pip

### Installation

```bash
# Cloner le dépôt
git clone https://github.com/berch-t/p06-air-paradis-api.git
cd p06-air-paradis-api

# Installer les dépendances
pip install -r requirements.txt

# Télécharger les ressources NLTK nécessaires
python -c "import nltk; nltk.download('punkt')"
```

### Lancement de l'API Flask

```bash
python application.py
```

L'API sera accessible à l'adresse http://localhost:5000

### Lancement de l'interface Streamlit

```bash
python run_streamlit.py
```

L'interface Streamlit sera accessible à l'adresse http://localhost:8501

## Endpoints API

- `GET /health` : Vérification de l'état de l'API
- `POST /predict` : Analyse du sentiment d'un tweet
  - Corps de la requête : `{"tweet": "Texte du tweet à analyser"}`
- `POST /feedback` : Envoi de feedback sur la prédiction
  - Corps de la requête : `{"tweet": "Texte du tweet", "predicted_sentiment": "Positif|Négatif", "is_correct": true|false}`

## Déploiement sur Azure

Cette application est conçue pour être facilement déployée sur Azure Web App:

1. Créez une ressource Azure Web App (Plan Linux, Python 3.9)
2. Configurez le déploiement continu depuis GitHub
3. Ajoutez une variable d'environnement `APPINSIGHTS_INSTRUMENTATION_KEY` si vous souhaitez utiliser Azure Application Insights

## Architecture du projet

- `api.py` : Contient l'API Flask et l'application Streamlit
- `application.py` : Point d'entrée pour Azure Web App
- `run_streamlit.py` : Script pour lancer l'interface Streamlit localement
- `requirements.txt` : Liste des dépendances
- `models/` : Dossier contenant le modèle BERT (créé automatiquement au premier lancement)

## Technologies utilisées

- **Flask** : Framework web léger pour l'API REST
- **Streamlit** : Framework pour l'interface utilisateur
- **BERT** (via Hugging Face Transformers) : Modèle de deep learning pour l'analyse de sentiment
- **Plotly** : Visualisations interactives
- **Azure Application Insights** : Monitoring et télémétrie
- **Pandas** : Manipulation des données
- **NLTK** : Prétraitement du texte

## Crédits

Développé par T.B - MIC (Marketing Intelligence Consulting) pour Air Paradis