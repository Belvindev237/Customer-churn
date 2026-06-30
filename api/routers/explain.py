from fastapi import APIRouter
from preprocessing import preprocess, explain_prediction
from schemas import ClientData

router = APIRouter(tags=["explain"])

@router.post("/explain")
def explain(client: ClientData):
    df = preprocess(client)
    shap_vals = explain_prediction(df)
    return {
        "feature_names": df.columns.tolist(),
        "shap_values": shap_vals.values[0].tolist(),
        "base_value": float(shap_vals.base_values[0].item())
    }