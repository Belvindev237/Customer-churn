import json
import joblib
import pandas as pd
import logging

from config import (
    SERVICE_COLS,
    COLS_OHE,
    COLS_STANDARD,
    ARTIFACTS,
    SEUIL
)

# Variables globales
model = None
scaler = None
ordinal_enc = None
feature_names = None

logger = logging.getLogger(__name__)

def load_artifacts():
    global model, scaler, ordinal_enc, feature_names
    model = joblib.load(ARTIFACTS["model"])
    scaler = joblib.load(ARTIFACTS["scaler"])
    ordinal_enc = joblib.load(ARTIFACTS["ordinal_enc"])
    with open(ARTIFACTS["feature_names"]) as f:
        feature_names = json.load(f)

def ensure_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Force la présence et l'ordre des 31 colonnes attendues."""
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]

def preprocess(data) -> pd.DataFrame:
    row = data.dict()

    # 1. Encodage binaire
    for col in ["Partner", "Dependents"]:
        row[col] = 1 if row[col] == "Yes" else 0
    row["TechSupport"] = 1 if row["TechSupport"] == "Yes" else 0

    # 2. Feature Engineering
    nb_services = sum(1 for col in SERVICE_COLS if row.get(col) == "Yes")
    row["NbServices"] = nb_services

    t = row["tenure"]
    if t <= 12: segment = "Nouveau"
    elif t <= 24: segment = "Junior"
    elif t <= 48: segment = "Etabli"
    else: segment = "Fidèle"
    row["TenureSegment"] = segment

    # 3. OrdinalEncoder corrigé (on passe un DataFrame nommé)
    df_seg = pd.DataFrame({'TenureSegment': [segment]})
    enc_val = float(ordinal_enc.transform(df_seg)[0][0])
    row["TenureSegment_enc"] = enc_val

    # 4. Contract One‑hot manuel
    row["Contract_One year"] = 1 if row["Contract"] == "One year" else 0
    row["Contract_Two year"] = 1 if row["Contract"] == "Two year" else 0

    # 5. DataFrame initial
    df = pd.DataFrame([row])
    df = df.drop(columns=["Contract", "TenureSegment"], errors="ignore")

    # 6. OneHotEncoding
    cols_to_encode = [c for c in COLS_OHE if c in df.columns]
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

    # 7. Alignement forcé (le correctif crucial)
    df = ensure_features(df, feature_names)

    # 8. Standardisation
    df[COLS_STANDARD] = scaler.transform(df[COLS_STANDARD])

    return df

def predict_churn(df: pd.DataFrame) -> dict:
    proba = float(model.predict_proba(df)[:, 1][0])
    churn = int(proba >= SEUIL)
    
    # Log utile pour déboguer
    logger.info(f"Prédiction effectuée, proba: {proba}")
    
    return {
        "churn_proba": round(proba, 4),
        "churn_prediction": churn,
        "label": "Churner ⚠️" if churn else "Fidèle ✅",
        "seuil_utilise": SEUIL,
    }