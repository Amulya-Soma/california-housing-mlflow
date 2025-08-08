from fastapi import FastAPI
from pydantic import BaseModel, Field
import mlflow.pyfunc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="California Housing Price Predictor")

# Define input data schema with Pydantic
class HousingFeatures(BaseModel):
    MedInc: float = Field(..., example=8.3252)
    HouseAge: float = Field(..., example=41.0)
    AveRooms: float = Field(..., example=6.9841)
    AveBedrms: float = Field(..., example=1.0238)
    Population: float = Field(..., example=322.0)
    AveOccup: float = Field(..., example=2.5556)
    Latitude: float = Field(..., example=37.88)
    Longitude: float = Field(..., example=-122.23)

# Load registered model from MLflow Model Registry
MODEL_NAME = "california-housing-model"
MODEL_VERSION = None  # Use latest version or specify a version

# Load model once on startup
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION or 'latest'}")

@app.post("/predict")
def predict(features: HousingFeatures):
    input_df = features.dict()
    logging.info(f"Received input: {input_df}")

    # MLflow pyfunc models expect DataFrame-like input
    import pandas as pd
    input_df = pd.DataFrame([input_df])

    # Predict
    prediction = model.predict(input_df)[0]
    logging.info(f"Prediction: {prediction}")

    return {"prediction": prediction}
