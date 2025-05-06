from fastapi import APIRouter
from api.schemas.input import FraudInput
import pandas as pd
from mlruns.model_loader import load_model  

router = APIRouter()
model = load_model()

@router.post("/")
def predict(input: FraudInput):
    try:
        data_dict = input.dict()
        df = pd.DataFrame([data_dict])
        prediction = model.predict(df)[0]
        label = "Fraud" if prediction == 1 else "Legitimate"
        return {"prediction": int(prediction), "message": label}
    except Exception as e:
        return {"error": str(e)}
