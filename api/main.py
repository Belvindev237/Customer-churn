# main.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
from config import SEUIL
from preprocessing import load_artifacts
from db import init_db
from routers import predict, explain, batch, stats


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    init_db()
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="Prédit la probabilité de churn d'un client Telco",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(batch.router)
app.include_router(stats.router)


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