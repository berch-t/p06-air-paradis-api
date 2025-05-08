# Air Paradis - API d'Analyse de Sentiment

Cette API permet d'analyser le sentiment (positif/négatif) de tweets concernant la compagnie aérienne Air Paradis.

## Fonctionnalités

- Prédiction du sentiment d'un tweet
- Endpoints pour les vérifications de santé de l'API
- Système de feedback pour améliorer le modèle
- Intégration avec Azure Application Insights pour le monitoring

## Endpoints

- `/health` : Vérification de l'état de l'API
- `/predict` : Analyse du sentiment d'un tweet
- `/feedback` : Envoi d'un feedback sur une prédiction

## Configuration

L'API utilise les variables d'environnement suivantes :
- `PORT` : Port sur lequel l'API s'exécute (par défaut: 5000)
- `APPINSIGHTS_INSTRUMENTATION_KEY` : Clé d'instrumentation pour Azure Application Insights

## Installation locale

```bash
# Cloner le dépôt
git clone [URL_DU_REPO]

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'API
python api.py
```

## Déploiement

Cette API est conçue pour être déployée sur Azure Web App. Voir la documentation de déploiement pour plus de détails.