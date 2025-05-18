"""
Script pour lancer directement l'application Streamlit
Cette version corrigée appelle directement l'application sans lancer plusieurs instances
"""
import os
import subprocess
import sys

# Définir la variable d'environnement pour indiquer à l'application qu'elle est exécutée en mode Streamlit
os.environ['STREAMLIT_RUN_MODE'] = 'true'

if __name__ == "__main__":
    # Chemin vers le fichier api.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_path = os.path.join(current_dir, "api.py")
    
    # Vérifier si le fichier existe
    if not os.path.exists(api_path):
        print(f"Erreur: Le fichier {api_path} n'existe pas!")
        sys.exit(1)
    
    # Lancer directement Streamlit avec api.py
    print(f"Lancement de Streamlit avec le fichier: {api_path}")
    subprocess.run(["streamlit", "run", api_path, "--server.port=8501"])