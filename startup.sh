#!/bin/bash

# Démarrer l'application Flask avec Gunicorn
gunicorn --bind=0.0.0.0 --timeout 600 application:app