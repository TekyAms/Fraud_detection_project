# 🛡️ NIBSS Fraud Detection Simulator

Application de **détection de fraude bancaire en temps réel** basée sur un modèle **XGBoost**, développée avec **Streamlit**. Le projet s'appuie sur des données synthétiques inspirées des dynamiques de la **NIBSS** (Nigerian Inter-Bank Settlement System).

![Badge Streamlit](https://img.shields.io/badge/Streamlit-1.57-FF4B4B?style=flat-square)
![Badge Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square)
![Badge XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange?style=flat-square)

---

## 📌 Aperçu

Ce projet permet de **simuler l'évaluation d'une transaction bancaire** : l'utilisateur renseigne les caractéristiques d'une transaction (montant, canal, fréquence, etc.) via une interface web intuitive, et le modèle **XGBoost** prédit en quelques millisecondes si la transaction est **frauduleuse ou légitime**, avec un niveau de probabilité détaillé.

À partir d'un jeu de **1 000 000 de transactions** (taux de fraude : 0,3 %), trois modèles ont été entraînés et comparés :

| Modèle | Précision (fraude) | Rappel (fraude) | Score F1 (fraude) |
|--------|--------------------|-----------------|-------------------|
| Logistic Regression | — | — | — |
| Random Forest | — | — | — |
| **XGBoost** ✅ | **0.77** | **0.82** | **0.79** |

> 💡 **Note** : les métriques détaillées de la Logistic Regression et du Random Forest sont disponibles dans le notebook `exploration.ipynb`.

---

## ✨ Fonctionnalités

- 🎛️ **Simulateur interactif** : réglage dynamique des paramètres de transaction via la barre latérale
- 📊 **Prédiction en temps réel** : probabilité de fraude (0–100 %) calculée par XGBoost
- 🚨 **Décision automatisée** : blocage de la transaction si suspicion élevée, avec recommandations système
- 🧠 **Pré-traitement identique à l'entraînement** : standardisation (`StandardScaler`), encodage one-hot, ingénierie de features (`amount_log`, `velocity_score`, `composite_risk`, etc.)
- 🌍 **Déployable sur Streamlit Cloud** : gratuit et sans frais d'infrastructure

---

## 🗂️ Structure du projet

```
Fraud_detection_project/
├── app.py                    # Application Streamlit (interface de simulation)
├── main.py                   # Script d'exemple (généré par défaut, non utilisé)
├── exploration.ipynb         # Analyse exploratoire + entraînement & évaluation des modèles
├── requirements.txt          # Dépendances Python minimales pour l'application
├── data/
│   ├── nibss_fraud_dataset.csv   # ⚠️ 1M transactions (~423 MB) — non versionné (voir .gitignore)
│   └── data_dictionary.csv       # 📖 Dictionnaire des 39 colonnes du jeu de données
└── models/
    ├── xgboost_fraud_model.pkl   # ✅ Modèle XGBoost entraîné et sérialisé
    ├── scaler.pkl                # StandardScaler ajusté sur les données d'entraînement
    └── feature_columns.pkl       # Ordre exact des colonnes attendu par le modèle
```

---

## 📊 Jeu de données

Le jeu de données (`data/nibss_fraud_dataset.csv`) contient **1 000 000 de transactions** sur l'année 2023, avec **39 colonnes** :

- **Identifiants & temporelles** : `transaction_id`, `customer_id`, `timestamp`, `hour`, `day_of_week`, `month`, `is_weekend`, `is_peak_hour`
- **Transaction** : `amount`, `channel` (Mobile, Web, POS, IB, ECOM, ATM), `merchant_category`, `bank`, `location`
- **Comportementales** : `tx_count_24h`, `amount_sum_24h`, `amount_mean_7d`, `amount_vs_mean_ratio`, `velocity_score`, `composite_risk`, …
- **Engeneered** : encodage cyclique (`hour_sin/cos`, `day_sin/cos`, `month_sin/cos`), `amount_log`, `amount_rounded`, `merchant_risk_score`
- **Cible** : `is_fraud` (0 = légitime, 1 = frauduleux) et `fraud_technique`

> ⚠️ **Attention** : le fichier CSV (~423 MB) **dépasse la limite de GitHub** (100 MB). Il est donc exclu du dépôt via `.gitignore`. Pour relancer le notebook, régénérez/téléchargez le CSV dans le dossier `data/`. Le **déploiement Streamlit ne nécessite pas ce fichier**, uniquement les modèles pré-entraînés.

---

## 🚀 Installation et exécution locale

### Prérequis
- Python **3.12+**
- Git

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/<votre-utilisateur>/Fraud_detection_project.git
cd Fraud_detection_project

# 2. Créer un environnement virtuel
python -m venv .venv

# 3. Activer l'environnement
# Windows :
.venv\Scripts\activate
# macOS / Linux :
source .venv/bin/activate

# 4. Installer les dépendances
pip install -r requirements.txt

# 5. Lancer l'application
streamlit run app.py
```

L'application s'ouvre automatiquement sur **http://localhost:8501**.

---

## ☁️ Déploiement sur Streamlit Cloud (gratuit)

### Depuis votre ordinateur

1. Poussez ce dépôt sur GitHub (voir section suivante).
2. Rendez-vous sur **https://share.streamlit.io**.
3. Cliquez sur **« Create app »** puis **« Deploy now »**.
4. Renseignez :
   - **Repository** : `votre-utilisateur/Fraud_detection_project`
   - **Branch** : `main`
   - **Main file path** : `app.py`
5. Cliquez sur **Deploy**. (⚠️ Un compte Streamlit Cloud est requis — connexion avec GitHub.)
6. Votre application est en ligne à une URL du type `https://votre-utilisateur-fraud-detection-project-xxxxx.streamlit.app`.

### Vérifications importantes

- ✅ Les modèles `.pkl` (~395 KB) sont **bien inclus** dans le dépôt (ils sont nécessaires à l'app).
- ✅ Le CSV de ~423 MB est **exclu** du dépôt — l'application n'en a pas besoin.
- ✅ `requirements.txt` est minimisé (5 packages seulement) pour un déploiement rapide et fiable.

---

## 🔍 Comment fonctionne la prédiction

1. L'utilisateur renseigne **montant**, **velocity score**, **ratio montant/moyenne**, **composite risk**, **canal** et **catégorie commerçant**.
2. L'app reconstruit le vecteur de features complet (mêmes transformations qu'à l'entraînement : `log1p`, one-hot encoding, valeurs par défaut pour les variables temporelles).
3. Les features sont **standardisées** via le `StandardScaler` pré-entraîné.
4. Le modèle **XGBoost** retourne la probabilité de fraude.
5. Si la probabilité dépasse le seuil (0,5), la transaction est **signalée comme frauduleuse** avec une recommandation de blocage.

---

## 🛠️ Dépendances

| Package | Version | Utilisation |
|---------|---------|-------------|
| `streamlit` | 1.57 | Interface web |
| `pandas` | 3.0 | Manipulation de données |
| `numpy` | 2.4 | Calculs numériques |
| `scikit-learn` | 1.8 | StandardScaler |
| `xgboost` | 3.2 | Modèle de classification |

---

## 📜 Licence

Projet réalisé à des fins **pédagogiques et de démonstration**. Les données sont **synthétiques** — toute ressemblance avec des transactions réelles est fortuite.

---

## 📧 Contact

Projet développé avec ❤️ — retours et suggestions bienvenus (GitHub Issues).