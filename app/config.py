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
    "model":      "model.pkl",
    "scaler":     "scaler.pkl",
    "ordinal_enc": "ordinal_encoder.pkl",
    "feature_names": "feature_names.json",
}