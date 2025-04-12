from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load model and encoder
model = joblib.load("model.pkl")
risk_encoder = joblib.load("risk_label_encoder.pkl")

# Dummy mapping (update with actual values if different)
film_type_mapping = {'PBAT': 0, 'PLA': 1, 'Starch Blend': 2}
soil_type_mapping = {'Loamy': 0, 'Sandy': 1, 'Clay': 2}

class InputData(BaseModel):
    film_type: str
    soil_ph: float
    soil_type: str
    moisture: float
    temperature: float
    uv_exposure: float
    duration: int

@app.post("/predict")
def predict(data: InputData):
    try:
        # Prepare input for model
        input_df = pd.DataFrame([{
            'Film_Type': film_type_mapping[data.film_type],
            'Soil_pH': data.soil_ph,
            'Soil_Type': soil_type_mapping[data.soil_type],
            'Moisture_Level': data.moisture,
            'Temperature': data.temperature,
            'UV_Exposure': data.uv_exposure,
            'Duration': data.duration
        }])

        # Make prediction
        pred = model.predict(input_df)[0]

        # Dummy degradation calculation (replace with real if needed)
        degradation_percent = round(100 - (data.duration * 0.3), 2)
        degradation_percent = max(0, min(100, degradation_percent))

        # Decode risk level
        risk_label = risk_encoder.inverse_transform([pred])[0]

        return {
            "degradation_percent": degradation_percent,
            "risk_level": risk_label
        }
    except Exception as e:
        return {"error": str(e)}
