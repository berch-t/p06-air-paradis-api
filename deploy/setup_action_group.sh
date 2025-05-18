#!/bin/bash

# Script pour configurer un groupe d'action (Action Group) dans Azure
# Ce groupe d'action envoie des notifications par email en cas d'erreurs consécutives

# Configuration
RESOURCE_GROUP="$1"
ACTION_GROUP_NAME="ModelErrorNotifications"
EMAIL_RECIPIENT="berchet.thomas@gmail.com"
SHORT_NAME="ModelAlerts"

# Vérification des arguments
if [ -z "$RESOURCE_GROUP" ]; then
    echo "Erreur: Groupe de ressources non spécifié"
    echo "Usage: $0 <resource_group_name>"
    exit 1
fi

echo "Configuration du groupe d'action pour les notifications par email..."

# Création du groupe d'action
az monitor action-group create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ACTION_GROUP_NAME" \
    --short-name "$SHORT_NAME" \
    --email-receiver "$SHORT_NAME" "$EMAIL_RECIPIENT"

# Récupération de l'ID du groupe d'action
ACTION_GROUP_ID=$(az monitor action-group show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ACTION_GROUP_NAME" \
    --query id -o tsv)

echo "Groupe d'action créé avec succès!"
echo "ID du groupe d'action: $ACTION_GROUP_ID"
echo ""
echo "Instructions pour GitHub Actions:"
echo "1. Ajoutez le secret suivant à votre dépôt GitHub:"
echo "   ACTION_GROUP_ID: $ACTION_GROUP_ID"
echo ""
echo "Configuration terminée!" 