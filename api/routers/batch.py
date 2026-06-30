from fastapi import APIRouter, UploadFile, HTTPException, Depends
from sqlalchemy.orm import Session
import pandas as pd
import io
from preprocessing import preprocess, predict_churn
from schemas import ClientData
from db import get_db, Prediction

router = APIRouter(tags=["batch"])

@router.post("/predict/batch")
async def predict_batch(file: UploadFile, db: Session = Depends(get_db)):
    try:
        content = await file.read()
        df_raw = pd.read_csv(io.BytesIO(content))

        results = []
        for _, row in df_raw.iterrows():
            client = ClientData(**row.to_dict())
            df = preprocess(client)
            result = predict_churn(df)

            db.add(Prediction(
                client_id=str(row.get("client_id", "")),
                churn_proba=result["churn_proba"],
                label=result["label"],
                monthly_charges=row.get("MonthlyCharges"),
            ))
            results.append({**row.to_dict(), **result})

        db.commit()
        return {"count": len(results), "results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))