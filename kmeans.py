import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import joblib

# Masquer les éléments inutiles de Streamlit
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stApp {margin: 0; padding: 0;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def main():
    # === 1. Chargement des données ===
    try:
        df = pd.read_csv('clustering_carte_credit.csv', sep=";", encoding='latin-1')
    except FileNotFoundError:
        st.error("❌ Fichier 'clustering_carte_credit.csv' non trouvé.")
        return

    # === 2. Colonnes utilisées pour la prédiction ===
    features = ['Age', 'salaire', 'Frequence_Paiements', 'Total_des_cheques']
    if not all(f in df.columns for f in features + ['segment_carte_credit']):
        st.error("❌ Colonnes manquantes dans le fichier.")
        return

    # === 3. Préparation des données ===
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=features)
    df_scaled['segment'] = df['segment_carte_credit']
    centroids = df_scaled.groupby('segment').mean().values
    segments_labels = df_scaled.groupby('segment').mean().index.tolist()

    # === 4. Détection des valeurs aberrantes ===
    limits = {}
    for feature in features:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        limits[feature] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    # === FORMULAIRE SIMPLIFIÉ ===
    with st.form("prediction_form"):
        st.write("**🧾 Informations du client**")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Âge", min_value=18, max_value=100)
            salaire = st.number_input("Salaire", min_value=450, max_value=10000)
        
        with col2:
            frequence = st.number_input("Fréquence des paiements", min_value=1, max_value=1000)
            total_cheques = st.number_input("Total des chèques", min_value=1, max_value=100000)
        
        submitted = st.form_submit_button("📊 Prédire le segment")
        
        if submitted:
            new_data = {
                'Age': age,
                'salaire': salaire,
                'Frequence_Paiements': frequence,
                'Total_des_cheques': total_cheques
            }
            
            anomalies = [f"{k} ({v})" for k, v in new_data.items() if not (limits[k][0] <= v <= limits[k][1])]
            
            if anomalies:
                st.warning(f"❌ Valeurs anormales: {', '.join(anomalies)}")
            else:
                new_client = np.array([[age, salaire, frequence, total_cheques]])
                new_client_scaled = scaler.transform(new_client)
                distances = cdist(new_client_scaled, centroids)
                closest = distances.argmin()
                predicted_segment = segments_labels[closest]
                st.success(f"✅ Segment estimé: **{predicted_segment}**")

if __name__ == "__main__":
    main()