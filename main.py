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
    model = joblib.load('model/best_model.pkl')  # Example model path
    scaler = joblib.load('scaler.pkl')  # Example scaler path
    logger.info("Model and scaler loaded successfully.")
except FileNotFoundError as e:
    logger.error(
        "Model or scaler file not found. Ensure 'best_model.pkl' and 'scaler.pkl' are in the correct location.")
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
    allow_origins=["*"],  # Allows all origins; restrict to specific URLs for production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Define the input schema using Pydantic
class InputData(BaseModel):
    Age: int = Field(..., ge=0, le=100, description="Age of the student (0-100)")
    Stress_Level: float = Field(..., ge=0.0, le=10.0, description="Stress level (0.0 to 10.0)")
    Depression_Score: float = Field(..., ge=0.0, le=10.0, description="Depression score (0.0 to 10.0)")
    Anxiety_Score: float = Field(..., ge=0.0, le=10.0, description="Anxiety score (0.0 to 10.0)")
    Physical_Activity: int = Field(..., ge=0, le=2, description="Physical activity level (0=Low, 1=Moderate, 2=High)")

    class Config:
        json_schema_extra = {  # Updated for Pydantic v2
            "example": {
                "Age": 25,
                "Stress_Level": 7.5,
                "Depression_Score": 6.0,
                "Anxiety_Score": 4.2,
                "Physical_Activity": 1
            }
        }

# Interpretation function: Modifying it to account for the scaled predictions
def interpret_sleep_quality(prediction, input_data):
    """
    Converts numeric sleep quality prediction into a meaningful label.
    Adds additional logic for forced "Poor" prediction based on certain feature conditions.
    """
    # If stress, depression, anxiety are high (close to 10) and physical activity is low, force "Poor"
    if input_data.Stress_Level >= 7 and input_data.Depression_Score >= 7 and input_data.Anxiety_Score >= 7 and input_data.Physical_Activity == 0:
        return "Poor"

    # Adjusting the thresholds based on the continuous prediction range (0.0 to 1.0)
    if prediction < 0.3:
        return "Poor"
    elif prediction < 0.7:
        return "Average"
    else:
        return "Good"

# Root endpoint
@app.get("/")
def read_root():
    """
    Root endpoint that provides basic API information.
    """
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the Mental Health Predictor API. Visit /docs for Swagger UI."}

# Prediction endpoint
@app.post("/predict", tags=["Prediction"])
def predict(input_data: InputData):
    """
    Predicts sleep quality based on mental health and activity data.
    """
    logger.info(f"Received input data: {input_data.dict()}")
    try:
        # Prepare the input data
        data = np.array([[input_data.Age, input_data.Stress_Level, input_data.Depression_Score,
                          input_data.Anxiety_Score, input_data.Physical_Activity]])

        # Scale the input data
        scaled_data = scaler.transform(data)
        logger.info(f"Scaled data: {scaled_data}")

        # Make the prediction
        prediction = model.predict(scaled_data)[0]
        logger.info(f"Prediction result: {prediction}")

        # Interpret the numeric prediction
        interpretation = interpret_sleep_quality(prediction, input_data)

        # Return the result with only the interpretation (without the encoded numeric value)
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