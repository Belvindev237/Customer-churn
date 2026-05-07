# main.py

from fastapi import FastAPI, HTTPException
from config import SEUIL, ARTIFACTS
from contextlib import asynccontextmanager
from preprocessing import load_artifacts, preprocess, predict_churn,explain_prediction
from schemas import ClientData


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="Prédit la probabilité de churn d'un client Telco",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "message": "Churn Prediction API — OK",
        "docs": "/docs",
        "seuil": SEUIL
    }


@app.get("/health")
def health():
    from preprocessing import model, feature_names
    return {
        "status": "ok",
        "model_type": type(model).__name__,
        "nb_features": len(feature_names),
        "seuil": SEUIL,
    }


@app.post("/predict")
def predict(client: ClientData):
    try:
        df = preprocess(client)
        result = predict_churn(df)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/explain")
def explain(client:ClientData):
    df=preprocess(client)
    shap_vals=explain_prediction(df)
    return {
        "feature_names":df.columns.tolist(),
        "shap_values": shap_vals.values[0].tolist(),
        "base_value": float(shap_vals.base_values[0].item())

    }