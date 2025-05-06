from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import mlflow.pyfunc

# Define FastAPI app
app = FastAPI()

# Load model from MLflow model registry (Production stage)
model = mlflow.pyfunc.load_model("models:/fraud-xgb/Production")

# Input schema
class Features(BaseModel):
    features: list[float]

@app.post("/predict")
async def predict(input_data: Features):
    if len(input_data.features) != 29:
        raise HTTPException(status_code=400, detail="Input should be a list of 29 features.")
    
    X = np.array(input_data.features).reshape(1, -1)
    prediction = model.predict(X)
    return {"prediction": int(prediction[0])}
