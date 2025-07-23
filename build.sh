#!/bin/bash
# Installer une version compatible de Python via pyenv (optionnel selon Render)
pyenv install 3.10.13
pyenv global 3.10.13

# Installer les dépendances
pip install -r requirements.txt
