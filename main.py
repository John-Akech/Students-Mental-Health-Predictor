from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MentalHealthPredictorAPI")

# Load the pre-trained model and scaler (update these paths as needed)
try:
    model = joblib.load('model/best_model.pkl')  # Ensure the correct path for your model
    scaler = joblib.load('scaler.pkl')  # Ensure the correct path for your scaler
    logger.info("Model and scaler loaded successfully.")
except FileNotFoundError as e:
    logger.error("Model or scaler file not found. Ensure 'best_model.pkl' and 'scaler.pkl' are in the correct location.")
    raise RuntimeError("Model or scaler file not found.")

# Create FastAPI app
app = FastAPI(
    title="Mental Health Predictor API",
    description="This API predicts sleep quality based on mental health and activity factors.",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to restrict origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input schema using Pydantic
class InputData(BaseModel):
    Age: int = Field(..., ge=0, le=100, description="Age of the student (0-100)")
    Stress_Level: float = Field(..., ge=0.0, le=10.0, description="Stress level (0.0 to 10.0)")
    Depression_Score: float = Field(..., ge=0.0, le=10.0, description="Depression score (0.0 to 10.0)")
    Anxiety_Score: float = Field(..., ge=0.0, le=10.0, description="Anxiety score (0.0 to 10.0)")
    Physical_Activity: int = Field(..., ge=0, le=2, description="Physical activity level (0=Low, 1=Moderate, 2=High)")

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 25,
                "Stress_Level": 7.5,
                "Depression_Score": 6.0,
                "Anxiety_Score": 4.2,
                "Physical_Activity": 1
            }
        }

# Interpretation function
def interpret_sleep_quality(prediction, input_data):
    if input_data.Stress_Level >= 7 and input_data.Depression_Score >= 7 and input_data.Anxiety_Score >= 7 and input_data.Physical_Activity == 0:
        return "Poor"
    if prediction < 0.3:
        return "Poor"
    elif prediction < 0.7:
        return "Average"
    else:
        return "Good"

# Root endpoint
@app.get("/")
def read_root():
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the Mental Health Predictor API. Visit /docs for Swagger UI."}

# Prediction endpoint
@app.post("/predict", tags=["Prediction"])
def predict(input_data: InputData):
    logger.info(f"Received input data: {input_data.dict()}")
    try:
        data = np.array([[input_data.Age, input_data.Stress_Level, input_data.Depression_Score,
                          input_data.Anxiety_Score, input_data.Physical_Activity]])
        scaled_data = scaler.transform(data)
        logger.info(f"Scaled data: {scaled_data}")

        prediction = model.predict(scaled_data)[0]
        logger.info(f"Prediction result: {prediction}")

        interpretation = interpret_sleep_quality(prediction, input_data)

        return {
            "message": f"Your predicted Sleep Quality is {interpretation}",
            "features": input_data.dict()
        }
    except ValueError as ve:
        logger.error(f"Data processing error: {str(ve)}")
        raise HTTPException(status_code=422, detail=f"Data processing error: {str(ve)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Run the application if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
