"""
API FastAPI — Prédiction du Churn Client
Auteur : Belvin Tsadjio
Date   : Avril 2026

Reproduce exactement le pipeline de preprocessing :
  1. Encodage binaire (Partner, Dependents, TechSupport)
  2. Feature Engineering (NbServices, TenureSegment)
  3. OrdinalEncoder sur TenureSegment
  4. OneHotEncoding (get_dummies) sur les variables catégorielles
  5. StandardScaler sur tenure, MonthlyCharges, TotalCharges
  6. Alignement des 31 colonnes
  7. Prédiction XGBoost avec seuil 0.35
"""

import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

# ── Chargement des artefacts ─────────────────────────────────────────────────
model          = joblib.load("model.pkl")
scaler         = joblib.load("scaler.pkl")
ordinal_enc    = joblib.load("ordinal_encoder.pkl")
feature_names  = json.load(open("feature_names.json"))

# ── Constantes ────────────────────────────────────────────────────────────────
SEUIL = 0.35

SERVICES_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies"
]

COLS_OHE = [
    "gender", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "StreamingTV", "StreamingMovies", "PaperlessBilling", "PaymentMethod"
]

COLS_STANDARD = ["tenure", "MonthlyCharges", "TotalCharges"]

# ── Schéma d'entrée (données brutes utilisateur) ──────────────────────────────
class ClientData(BaseModel):
    # Données démographiques
    gender          : Literal["Male", "Female"]
    SeniorCitizen   : Literal[0, 1]
    Partner         : Literal["Yes", "No"]
    Dependents      : Literal["Yes", "No"]

    # Contrat & facturation
    tenure          : int   = Field(..., ge=0, le=72,  description="Ancienneté en mois")
    Contract        : Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod   : Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    MonthlyCharges  : float = Field(..., ge=0, description="Charges mensuelles en €")
    TotalCharges    : float = Field(..., ge=0, description="Charges totales en €")

    # Services téléphoniques
    PhoneService    : Literal["Yes", "No"]
    MultipleLines   : Literal["Yes", "No", "No phone service"]

    # Services internet
    InternetService : Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity  : Literal["Yes", "No", "No internet service"]
    OnlineBackup    : Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport     : Literal["Yes", "No", "No internet service"]
    StreamingTV     : Literal["Yes", "No", "No internet service"]
    StreamingMovies : Literal["Yes", "No", "No internet service"]

    class Config:
        json_schema_extra = {
            "example": {
                "gender"          : "Female",
                "SeniorCitizen"   : 0,
                "Partner"         : "Yes",
                "Dependents"      : "No",
                "tenure"          : 5,
                "Contract"        : "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod"   : "Electronic check",
                "MonthlyCharges"  : 70.5,
                "TotalCharges"    : 352.5,
                "PhoneService"    : "Yes",
                "MultipleLines"   : "No",
                "InternetService" : "Fiber optic",
                "OnlineSecurity"  : "No",
                "OnlineBackup"    : "No",
                "DeviceProtection": "No",
                "TechSupport"     : "No",
                "StreamingTV"     : "No",
                "StreamingMovies" : "No"
            }
        }


def preprocess(data: ClientData) -> pd.DataFrame:
    """
    Reproduit exactement le pipeline du notebook preprocessing.
    """
    # 1. Dictionnaire brut
    row = data.model_dump()

    # 2. Encodage binaire (identique au preprocessing)
    for col in ["Partner", "Dependents"]:
        row[col] = 1 if row[col] == "Yes" else 0

    # TechSupport : "No internet service" → 0
    row["TechSupport"] = 1 if row["TechSupport"] == "Yes" else 0

    # 3. Feature Engineering
    #    NbServices : nombre de services à "Yes"
    nb_services = sum(
        1 for col in SERVICES_COLS
        if row.get(col) == "Yes"
    )
    row["NbServices"] = nb_services

    #    TenureSegment : segmentation par ancienneté
    t = row["tenure"]
    if t <= 12:
        segment = "Nouveau"
    elif t <= 24:
        segment = "Junior"
    elif t <= 48:
        segment = "Etabli"
    else:
        segment = "Fidèle"
    row["TenureSegment"] = segment

    # 4. Encodage OrdinalEncoder sur TenureSegment
    row["TenureSegment_enc"] = float(
        ordinal_enc.transform([[segment]])[0][0]
    )

    # 5. Encodage Contract (OHE manuel — identique à get_dummies drop_first)
    row["Contract_One year"] = 1 if row["Contract"] == "One year"  else 0
    row["Contract_Two year"] = 1 if row["Contract"] == "Two year"  else 0

    # 6. Création du DataFrame
    df = pd.DataFrame([row])

    # 7. Suppression des colonnes non utilisées par le modèle
    df = df.drop(columns=["Contract", "TenureSegment"], errors="ignore")

    # 8. OneHotEncoding des variables catégorielles restantes
    #    (identique à pd.get_dummies drop_first=True)
    cols_present = [c for c in COLS_OHE if c in df.columns]
    df = pd.get_dummies(df, columns=cols_present, drop_first=True)

    # 9. Alignement sur les 31 colonnes exactes du modèle
    df = df.reindex(columns=feature_names, fill_value=0)

    # 10. Standardisation (transform uniquement — jamais fit !)
    df[COLS_STANDARD] = scaler.transform(df[COLS_STANDARD])

    return df


# ── Application FastAPI ───────────────────────────────────────────────────────
app = FastAPI(
    title       ="Churn Prediction API",
    description ="Prédit la probabilité de churn d'un client Telco",
    version     ="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Churn Prediction API — OK",
        "docs"   : "/docs",
        "seuil"  : SEUIL
    }


@app.get("/health")
def health():
    """Vérification que l'API et les artefacts sont bien chargés."""
    return {
        "status"       : "ok",
        "model"        : type(model).__name__,
        "nb_features"  : len(feature_names),
        "seuil"        : SEUIL
    }


@app.post("/predict")
def predict(client: ClientData):
    """
    Prédit si un client va churner.

    Retourne :
    - churn_proba     : probabilité de churn (0 à 1)
    - churn_prediction: 1 = Churner, 0 = Fidèle
    - label           : 'Churner ⚠️' ou 'Fidèle ✅'
    - seuil_utilise   : seuil de décision appliqué (0.35)
    """
    try:
        df    = preprocess(client)
        proba = float(model.predict_proba(df)[:, 1][0])
        churn = int(proba >= SEUIL)

        return {
            "churn_proba"     : round(proba, 4),
            "churn_prediction": churn,
            "label"           : "Churner ⚠️" if churn else "Fidèle ✅",
            "seuil_utilise"   : SEUIL
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))