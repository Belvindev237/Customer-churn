# config.py

SEUIL = 0.35

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies",
]

COLS_OHE = [
    "gender", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "StreamingTV", "StreamingMovies", "PaperlessBilling", "PaymentMethod",
]

COLS_STANDARD = ["tenure", "MonthlyCharges", "TotalCharges"]


ARTIFACTS = {
    "model": "models/model.pkl",
    "scaler": "models/scaler.pkl",
    "ordinal_enc": "models/ordinal_encoder.pkl",
    "feature_names": "models/feature_names.json",
}