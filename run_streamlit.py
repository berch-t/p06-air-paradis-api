import os
import sys
import streamlit.web.bootstrap

# Définir la variable d'environnement pour indiquer à l'application qu'elle est exécutée en mode Streamlit
os.environ['STREAMLIT_RUN_MODE'] = 'true'

# Lancer l'application Streamlit
if __name__ == '__main__':
    # Exécuter l'application Streamlit
    sys.argv = ["streamlit", "run", "api.py"]
    streamlit.web.bootstrap.run()