from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from preprocessing import preprocess, predict_churn
from schemas import ClientData
from db import get_db, Prediction

router = APIRouter(tags=["predict"])

@router.post("/predict")
def predict(client: ClientData, db: Session = Depends(get_db)):
    try:
        df = preprocess(client)
        result = predict_churn(df)

        db.add(Prediction(
            churn_proba=result["churn_proba"],
            label=result["label"],
            monthly_charges=client.MonthlyCharges,
        ))
        db.commit()

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))