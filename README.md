# 🎬 Projet : Prédiction de l'impact culturel des séries afrodescendantes

## 📖 Description

Ce projet est un **MVP** qui combine **data science, NLP et APIs** pour prédire si une série ou un film est susceptible de plaire au **public afrodescendant**.  

Il s’appuie sur :  
- Des données enrichies issues de **TMDB API**, **Reddit API**, **Wikipedia**, et **Ethnicelebs**.  
- Un modèle de **régression logistique explicable** (XAI-friendly) pour évaluer l’impact des variables.  
- Un pipeline complet qui va de la **collecte de données** jusqu’à la **prédiction automatisée**.  

---

## ⚙️ Fonctionnalités

- 📊 **Préparation des données** (`ml.py`)  
  - Encodage des variables catégorielles.  
  - Analyse des corrélations avec la variable cible.  
  - Entraînement d’un modèle de régression logistique.  
  - Évaluation (classification report, confusion matrix, calibration).  

- 🌍 **Collecte & enrichissement** (`predict_series.py`)  
  - Récupération des informations depuis TMDB (casting, créateurs, genres, pays, plateformes).  
  - Détection de l’ascendance afrodescendante (Wikipedia + NLP + Ethnicelebs).  
  - Estimation de la popularité sociale via Reddit.  
  - Calcul du **taux de complétion estimé**.  
  - Génération automatique d’un **vecteur de features** pour prédiction.  

- 🤖 **Prédiction**  
  - Sauvegarde/chargement du modèle entraîné (`logistic_model.pkl`).  
  - Préparation des nouvelles séries/films.  
  - Prédiction de la probabilité de plaire au public afrodescendant.  

---

## 📂 Structure du projet

```
├── series_afrodescendants.xlsx   # Dataset (synthetique + données réelles enrichies)
├── ml.py                         # Script d'entraînement et d'analyse du modèle
├── predict_series.py             # Script principal de collecte et prédiction
├── logistic_model.pkl            # Modèle sauvegardé
├── requirements.txt              # Dépendances Python
└── README.md                     # Documentation
```

---

## 🚀 Installation

1. **Cloner le projet**
   ```bash
   git clone https://github.com/<TON_USER>/<TON_REPO>.git
   cd <TON_REPO>
   ```

2. **Créer un environnement virtuel**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / Mac
   venv\Scripts\activate      # Windows
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔑 Configuration

- Créer un compte et obtenir une clé API sur :  
  - [TheMovieDB (TMDB)](https://www.themoviedb.org/documentation/api)  
  - [Reddit (PRAW)](https://www.reddit.com/prefs/apps)  

- Ajouter vos clés dans `predict_series.py` :
  ```python
  API_KEY = "VOTRE_TMDB_API_KEY"
  REDDIT_CLIENT_ID = "VOTRE_CLIENT_ID"
  REDDIT_CLIENT_SECRET = "VOTRE_SECRET"
  REDDIT_USER_AGENT = "python:NomProjet:v1.0 (by u/votre_user)"
  ```

---

## ▶️ Utilisation

### 1. Entraîner le modèle
```bash
python ml.py
```
➡️ Génère `logistic_model.pkl`

### 2. Faire une prédiction
```bash
python predict_series.py
```
➡️ Saisir l’ID TMDB et la catégorie (`movie` ou `tv`).  
➡️ Le script affiche les résultats et la probabilité prédite.

---

## 📊 Exemple de sortie

```
Titre : Moonlight
Pourcentage afrodescendant (acteurs) : 65.0%
Createur_afrodescendant : Oui
Popularité réseaux estimée (Reddit) : Moyenne (score: 1582000)
Taux de completion calculé : 78.5%
---
Prédiction : Oui
Probabilité de plaire au public afrodescendant : 84.3%
```

---

## 📌 Limitations

⚠️ Le dataset d’entraînement contient une partie **synthétique générée via IA (Grok)** faute de données accessibles.  
Ce projet est un **MVP expérimental** et ne prétend pas fournir des prédictions définitives, mais explorer les usages de l’IA pour les questions de **diversité et inclusion**.

---

## 📜 Licence

Projet Open Source
