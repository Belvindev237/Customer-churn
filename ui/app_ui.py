import streamlit as st
import requests
import os

# URL de l'API (Dynamique : "http://api:8000/predict" dans Docker, sinon localhost)
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.set_page_config(page_title="Churn Prediction", layout="centered")
st.title("📊 Churn Prediction Interface")

with st.form("churn_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Genre", ["Male", "Female"])
        tenure = st.number_input("Ancienneté (mois)", 0, 72, 1)
        monthly_charges = st.number_input("Charges mensuelles (€)", 0.0, 200.0, 50.0)
        contract = st.selectbox("Contrat", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Mode de paiement", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    with col2:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        tech_support = st.selectbox("Support Technique", ["Yes", "No", "No internet service"])
        paperless = st.selectbox("Facturation dématérialisée", ["Yes", "No"])
        partner = st.selectbox("Partenaire", ["Yes", "No"])
        dependents = st.selectbox("Personnes à charge", ["Yes", "No"])

    submit = st.form_submit_button("Calculer le risque de churn")

if submit:
    # Construire le payload (assure-toi d'inclure TOUS les champs du modèle)
    data = {
        "gender": gender, "SeniorCitizen": 0, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges,
        "TotalCharges": monthly_charges * tenure, # Simplifié pour l'exemple
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": internet,
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": tech_support, "StreamingTV": "No", "StreamingMovies": "No"
    }

    try:
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            res = response.json()
            st.metric("Probabilité", f"{res['churn_proba']:.1%}")
            if res['churn_prediction'] == 1:
                st.error(f"Statut: {res['label']}")
            else:
                st.success(f"Statut: {res['label']}")
        else:
            st.error(f"Erreur API ({response.status_code}): {response.text}")
    except Exception as e:
        st.error(f"Connexion impossible: {e}")