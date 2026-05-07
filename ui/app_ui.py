import streamlit as st
import requests
import os
import pandas as pd

#API_URL = os.getenv("API_URL", "http://localhost:8000")
API_URL = "http://api:8000"

st.set_page_config(page_title="Churn Prediction", layout="centered")
st.title("📊 Churn Prediction Interface")

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Genre", ["Male", "Female"])
        tenure = st.number_input("Ancienneté (mois)", 0, 72, 1)
        monthly_charges = st.number_input("Charges mensuelles (€)", 0.0, 200.0, 50.0)
        contract = st.selectbox("Contrat", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox(
            "Mode de paiement",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )

    with col2:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        tech_support = st.selectbox("Support Technique", ["Yes", "No", "No internet service"])
        paperless = st.selectbox("Facturation dématérialisée", ["Yes", "No"])
        partner = st.selectbox("Partenaire", ["Yes", "No"])
        dependents = st.selectbox("Personnes à charge", ["Yes", "No"])

    submit = st.form_submit_button("Calculer le risque de churn")

if submit:
    input_data = {
        "gender": gender,
        "SeniorCitizen": 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": monthly_charges * tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": internet,
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": tech_support,
        "StreamingTV": "No",
        "StreamingMovies": "No"
    }

    st.session_state["input_data"] = input_data

    try:
        response = requests.post(f"{API_URL}/predict", json=input_data)
        if response.status_code == 200:
            res = response.json()
            st.session_state["last_prediction"] = res

            st.metric("Probabilité", f"{res['churn_proba']:.1%}")
            if res["churn_prediction"] == 1:
                st.error(f"Statut: {res['label']}")
            else:
                st.success(f"Statut: {res['label']}")
        else:
            st.error(f"Erreur API ({response.status_code}): {response.text}")
    except Exception as e:
        st.error(f"Connexion impossible: {e}")

if "input_data" in st.session_state and st.button("Explain prediction"):
    try:
        response = requests.post(f"{API_URL}/explain", json=st.session_state["input_data"])
        if response.status_code == 200:
            res = response.json()
            df_shap = pd.DataFrame({
                "feature": res["feature_names"],
                "shap": res["shap_values"]
            }).sort_values("shap", key=abs, ascending=False)

            st.bar_chart(df_shap.set_index("feature")["shap"])
        else:
            st.error(f"Erreur API: {response.text}")
    except Exception as e:
        st.error(f"Connexion impossible: {e}")