import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Configuration de la page Streamlit
st.set_page_config(
    page_title="NIBSS Fraud Detection Simulator",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Simulateur de Détection de Fraude Bancaire (Modèle XGBoost)")
st.write("Données basées sur les dynamiques de la NIBSS (Nigerian Interbank Settlement System)")

# 2. Chargement des artefacts du modèle
@st.cache_resource
def load_model_artifacts():
    with open("models/xgboost_fraud_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/feature_columns.pkl", "rb") as f:
        columns = pickle.load(f)
    return model, scaler, columns

try:
    model_xgb, scaler, feature_columns = load_model_artifacts()
except FileNotFoundError:
    st.error(" Erreur : Les fichiers du modèle sont introuvables. Exécutez d'abord la sauvegarde dans votre notebook.")
    st.stop()

# 3. Formulaire de saisie utilisateur (Interface Web)
st.sidebar.header("Paramètres de la Transaction")

# Variables numériques majeures
amount = st.sidebar.number_input("Montant de la transaction (Naira - NGN)", min_value=1.0, value=50000.0, step=500.0)
velocity_score = st.sidebar.slider("Velocity Score (Fréquence sur 24h)", 0.0, 10.0, 0.5, step=0.1)
amount_vs_mean_ratio = st.sidebar.slider("Montant actuel vs Moyenne Historique", 0.1, 50.0, 1.0, step=0.1)
composite_risk = st.sidebar.slider("Composite Risk Score (Score Global)", 0.0, 1.0, 0.2, step=0.05)

# Variables catégorielles (Sélection)
channel = st.sidebar.selectbox("Canal de transaction", ["Web", "Mobile", "POS", "IB", "ECOM", "ATM"])
merchant_category = st.sidebar.selectbox("Catégorie du Commerçant", ["Grocery", "Entertainment", "Transport", "Fuel", "Transfer", "Retail"])

# Variables temporelles (contribuent à l'encodage cyclique)
hour = st.sidebar.slider("Heure de la transaction", 0, 23, 12)
day_of_week = st.sidebar.selectbox("Jour de la semaine", ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"], index=3)
month = st.sidebar.slider("Mois de la transaction", 1, 12, 6)

# 4. Reconstruction des features à la volée pour la prédiction
# On crée un dictionnaire vide aligné sur l'ordre EXACT de l'entraînement
base_data = {col: 0.0 for col in feature_columns}

# Mise à jour des valeurs saisies par l'utilisateur
base_data['amount'] = amount
base_data['amount_log'] = np.log1p(amount)
base_data['velocity_score'] = velocity_score
base_data['amount_vs_mean_ratio'] = amount_vs_mean_ratio
base_data['composite_risk'] = composite_risk

# Variables temporelles + encodage cyclique cohérent avec les valeurs choisies
day_index = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"].index(day_of_week)
base_data['hour'] = float(hour)
base_data['day_of_week'] = float(day_index)
base_data['month'] = float(month)

base_data['hour_sin'] = np.sin(2 * np.pi * hour / 24)
base_data['hour_cos'] = np.cos(2 * np.pi * hour / 24)
base_data['day_sin'] = np.sin(2 * np.pi * day_index / 7)
base_data['day_cos'] = np.cos(2 * np.pi * day_index / 7)
base_data['month_sin'] = np.sin(2 * np.pi * month / 12)
base_data['month_cos'] = np.cos(2 * np.pi * month / 12)

# Flags temporels dérivés des mêmes valeurs
base_data['is_weekend'] = 1.0 if day_index >= 5 else 0.0
base_data['is_peak_hour'] = 1.0 if hour in (10, 11, 14, 15, 16) else 0.0

# Gestion du One-Hot Encoding manuel pour la saisie (Exemple pour channel et merchant)
if f"channel_{channel}" in base_data:
    base_data[f"channel_{channel}"] = 1.0
if f"merchant_category_{merchant_category}" in base_data:
    base_data[f"merchant_category_{merchant_category}"] = 1.0

# Conversion en DataFrame avec le bon ordre de colonnes
input_df = pd.DataFrame([base_data])[feature_columns]

# 5. Standardisation et Prédiction
input_scaled = scaler.transform(input_df)
prediction = model_xgb.predict(input_scaled)[0]
prediction_proba = model_xgb.predict_proba(input_scaled)[0][1]

# 6. Affichage des Résultats à l'écran
st.subheader("Diagnostic de l'Incident de Fraude")

col1, col2 = st.columns(2)

with col1:
    st.metric(label="Probabilité de Fraude estimée par XGBoost", value=f"{prediction_proba * 100:.2f} %")

with col2:
    if prediction == 1:
        st.error("🚨 TRANSACTION BLOQUÉE : Suspicion Élevée de Fraude")
        st.markdown(
            "**Recommandation Système** : Déclenchement d'un gel temporaire du compte. "
            "Envoi d'une notification push pour vérification d'identité."
        )
    else:
        st.success("✅ TRANSACTION VALIDÉE : Comportement Sain")
        st.markdown("**Recommandation Système** : Autorisation de routage vers la banque partenaire NIBSS.")

# Section d'analyse contextuelle
st.info(
    f"💡 **Note d'analyse** : Vous avez sélectionné le canal **{channel}**. "
    f"Le modèle surveille particulièrement si le montant (actuellement {amount:,} NGN) "
    f"représente une rupture par rapport aux habitudes du client (Ratio actuel : {amount_vs_mean_ratio}x)."
)
