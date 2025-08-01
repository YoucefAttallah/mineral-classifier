# 🪨 Mineral Classifier Web App

Une application web simple basée sur **Flask** permettant de classer des images de minéraux à l’aide d’un modèle d’intelligence artificielle pré-entraîné (TensorFlow/Keras).

## 🔍 Fonctionnalités

- Interface utilisateur conviviale (HTML + Bootstrap)
- Téléversement d’image de spécimen minéral
- Prédiction automatique de la classe du minéral
- Déploiement en ligne (compatible Render, Heroku, etc.)

---

## 🚀 Démo en ligne

👉 [Lien vers l'application sur Render](https://mineral-classifier.onrender.com) *(à mettre après déploiement)*

---

## 📁 Structure du projet

mineral_app/
├── app.py # Application Flask
├── model.h5 # Modèle entraîné
├── requirements.txt # Dépendances Python
├── render.yaml # Fichier de configuration pour Render
├── templates/
│ └── index.html # Interface utilisateur
└── static/ # (facultatif) Dossier pour styles CSS ou images


---

## ⚙️ Installation locale

```bash
git clone https://github.com/ton-utilisateur/mineral_app.git
cd mineral_app

# Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate    # Sur Windows
source venv/bin/activate # Sur Linux/Mac

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
python app.py


☁️ Déploiement avec Render
Créer un compte sur https://render.com

Lier votre dépôt GitHub

Render détectera automatiquement le fichier render.yaml

Lancer le déploiement et accéder à l’URL publique

📦 Dépendances principales
Flask

TensorFlow / Keras

Gunicorn (pour Render)

Bootstrap (via CDN)

🧠 Modèle utilisé
Le modèle model.h5 est un réseau de neurones convolutif entraîné pour reconnaître plusieurs classes de minéraux à partir d’images.

🙌 Remerciements
Développé par Youcef Attallah dans le cadre d’un projet de classification minérale assistée par intelligence artificielle.#   m i n e r a l - c l a s s i f i e r 
 
 
