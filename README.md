# 🛡️ NIBSS Fraud Detection Simulator

Application de **détection de fraude bancaire en temps réel** basée sur un modèle **XGBoost**, développée avec **Streamlit**. Le projet s'appuie sur des données synthétiques inspirées des dynamiques de la **NIBSS** (Nigerian Inter-Bank Settlement System).

[![Badge Streamlit](https://img.shields.io/badge/Streamlit-1.57-FF4B4B?style=flat-square)](https://tekyamsfrauddetectionproject.streamlit.app/)
![Badge Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square)
![Badge XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange?style=flat-square)
[![Badge App Live](https://img.shields.io/badge/App_Streamlit-live-2ea44f?style=flat-square)](https://tekyamsfrauddetectionproject.streamlit.app/)

---

##  Aperçu

Ce projet permet de **simuler l'évaluation d'une transaction bancaire** : l'utilisateur renseigne les caractéristiques d'une transaction (montant, canal, fréquence, etc.) via une interface web intuitive, et le modèle **XGBoost** prédit en quelques millisecondes si la transaction est **frauduleuse ou légitime**, avec un niveau de probabilité détaillé.

À partir d'un jeu de **1 000 000 de transactions** (taux de fraude : 0,3 %), trois modèles ont été entraînés et comparés :

| Modèle | ROC-AUC | PR-AUC | Précision (fraude) | Rappel (fraude) | F1 (fraude) |
|--------|---------|--------|--------------------|-----------------|-------------|
| Logistic Regression | — | — | 0.01 | 0.70 | 0.01 |
| Random Forest | — | — | 0.01 | 0.78 | 0.03 |
| **XGBoost** ✅ | **0.969** | **0.839** | **0.77** | **0.82** | **0.79** |

**Optimisation du seuil de décision** : le seuil par défaut (0,5) n'est pas optimal. En ajustant le seuil à **0,65**, on obtient un **F1 = 0,854** (Précision = 0,945 / Rappel = 0,778), soit moins de 1% de faux positifs.

**Validation croisée stratifiée (5 replis)** : ROC-AUC moyen = **0,959 ± 0,003**, Rappel moyen = **0,807 ± 0,010** performance stable et non dépendante d'un découpage chanceux.

>  **NB** : l'accuracy (1,00) est trompeuse sur un dataset déséquilibré à 99,7 % . Les vraies métriques de référence sont le **ROC-AUC**, la **PR-AUC**, le **rappel** (ne pas rater de fraude) et la **précision** (ne pas bloquer d'honnêtes clients). XGBoost excelle sur tous ces axes grâce à `scale_pos_weight=333`.

---

##  Fonctionnalités

-  **Simulateur interactif** : réglage dynamique des paramètres de transaction via la barre latérale
-  **Prédiction en temps réel** : probabilité de fraude (0–100 %) calculée par XGBoost
-  **Décision automatisée** : blocage de la transaction si suspicion élevée, avec recommandations système
-  **Pré-traitement identique à l'entraînement** : standardisation (`StandardScaler`), encodage one-hot, ingénierie de features (`amount_log`, `velocity_score`, `composite_risk`, etc.)
-  **Déploiement sur Streamlit Cloud** : gratuit et sans frais d'infrastructure

---

##  Structure du projet

```
Fraud_detection_project/
├── app.py                    # Application Streamlit (interface de simulation)
├── exploration.ipynb         # Analyse exploratoire + entraînement & évaluation des modèles
├── requirements.txt          # Dépendances Python minimales pour l'application
├── data/
│   ├── nibss_fraud_dataset.csv   #  1M transactions (~423 MB) 
│   └── data_dictionary.csv       #  Dictionnaire des 38 colonnes du jeu de données
└── models/
    ├── xgboost_fraud_model.pkl   #  Modèle XGBoost entraîné et sérialisé
    ├── scaler.pkl                # StandardScaler ajusté sur les données d'entraînement
    └── feature_columns.pkl       # Ordre exact des 63 colonnes attendu par le modèle
```

---

##  Jeu de données

Le jeu de données (`data/nibss_fraud_dataset.csv`) contient **1 000 000 de transactions** sur l'année 2023, avec **38 colonnes** :

- **Identifiants & temporelles** : `transaction_id`, `customer_id`, `timestamp`, `hour`, `day_of_week`, `month`, `is_weekend`, `is_peak_hour`
- **Transaction** : `amount`, `channel` (Mobile, Web, POS, IB, ECOM, ATM), `merchant_category`, `bank`, `location`
- **Comportementales** : `tx_count_24h`, `amount_sum_24h`, `amount_mean_7d`, `amount_vs_mean_ratio`, `velocity_score`, `composite_risk`, …
- **Engeneered** : encodage cyclique (`hour_sin/cos`, `day_sin/cos`, `month_sin/cos`), `amount_log`, `amount_rounded`, `merchant_risk_score`
- **Cible** : `is_fraud` (0 = légitime, 1 = frauduleux) et `fraud_technique`

> ⚠️ **Attention** : le fichier CSV (~423 MB) **dépasse la limite de GitHub** (100 MB). Il est donc exclu du dépôt via `.gitignore`. Pour relancer le notebook, régénérez/téléchargez le CSV dans le dossier `data/`. Le **déploiement Streamlit ne nécessite pas ce fichier**, uniquement les modèles pré-entraînés.

---

##  Évaluation et validation

Le notebook `exploration.ipynb` ne se limite pas à l'accuracy, il inclut une **évaluation robuste** adaptée au déséquilibre des classes :

- **ROC-AUC** (0,969) : capacité du modèle à distinguer fraudes et transactions légitimes ;
- **PR-AUC** (0,839) : pertinence des alertes, métrique clé quand la classe cible représente 0,3 % ;
- **Courbe Précision-Rappel** avec baseline non-informative ;
- **Optimisation du seuil de décision** : tableau des F1/Précision/Rappel pour chaque seuil (0,05 → 0,95), avec identification du seuil optimal (0,65) ;
- **Validation croisée stratifiée** (5 replis) : stabilité des performances entre replis ;
- **Split temporel anti-leakage** : entraînement sur le passé, test sur le futur, pour reproduire le déploiement réel.

---

##  Installation et exécution locale

### Prérequis
- Python **3.12+**
- Git

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/TekyAms/Fraud_detection_project.git
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

##  Comment fonctionne la prédiction

1. L'utilisateur renseigne **montant**, **velocity score**, **ratio montant/moyenne**, **composite risk**, **canal** et **catégorie commerçant**.
2. L'app reconstruit le vecteur de features complet (mêmes transformations qu'à l'entraînement : `log1p`, one-hot encoding, valeurs par défaut pour les variables temporelles).
3. Les features sont **standardisées** via le `StandardScaler` pré-entraîné.
4. Le modèle **XGBoost** retourne la probabilité de fraude.
5. Si la probabilité dépasse le seuil (0,5), la transaction est **signalée comme frauduleuse** avec une recommandation de blocage.

---

##  Dépendances

| Package | Version | Utilisation |
|---------|---------|-------------|
| `streamlit` | 1.57 | Interface web |
| `pandas` | 3.0 | Manipulation de données |
| `numpy` | 2.4 | Calculs numériques |
| `scikit-learn` | 1.8 | StandardScaler |
| `xgboost` | 3.2 | Modèle de classification |
| `matplotlib` | 3.10 | Graphiques (notebook) |
| `seaborn` | 0.13 | Visualisation statistique (notebook) |
| `shap` | 0.51 | Interprétabilité du modèle (notebook) |

---

##  Application en ligne

L'application est déployée gratuitement sur **Streamlit Cloud** et accessible à tout moment :

👉 **[tekyamsfrauddetectionproject.streamlit.app](https://tekyamsfrauddetectionproject.streamlit.app/)**

---

##  Contact

Projet développé par **Tèkiyath AMOUSSA alias TekyAms**, retours et suggestions bienvenus.