from fastapi import FastAPI, HTTPException
import logging
import os
import pickle
from pydantic import BaseModel, Field
import numpy as np
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load Model
MODEL_PATH = 'model/insurance_model.pkl'  # Ensure correct file path
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    logger.info("✅ Model loaded successfully.")
except FileNotFoundError:
    logger.error(f"❌ Model file '{MODEL_PATH}' not found!")
    raise RuntimeError(f"Model file '{MODEL_PATH}' not found!")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    raise RuntimeError("Model could not be loaded.")

# Initialize FastAPI
app = FastAPI(
    title="Medical Insurance Prediction API",
    description="Predict insurance charges based on age, BMI, number of children, and smoking status.",
    version="1.1.0"
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to restrict access if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Model
class PredictionReq(BaseModel):
    BMI: float = Field(gt=0, lt=50, default=18.5, description="Your BMI Index")
    Age: int = Field(gt=0, lt=120, description="Your Age")
    Children: int = Field(ge=0, le=10, default=0, description="Number of children")
    Smoker: str = Field(default="no", description="Smoker status (yes/no)")

    def get_smoker_value(self):
        if self.Smoker.lower() not in ["yes", "no"]:
            raise ValueError("Smoker must be 'yes' or 'no'")
        return 1 if self.Smoker.lower() == "yes" else 0

# Root Route
@app.get("/", tags=["General"])
def root():
    return {"message": "Welcome to the Medical Insurance Prediction API!"}

# Prediction Endpoint
@app.post("/predict", tags=["Prediction"], responses={200: {"description": "Prediction successful"}, 400: {"description": "Invalid input"}})
def predict(req: PredictionReq):
    try:
        logger.info(f"📩 Received Request: {req}")

        # Convert smoker status to numeric
        smoker_value = req.get_smoker_value()

        # Prepare input as a NumPy array
        input_data = np.array([[req.Age, req.BMI, req.Children, smoker_value]], dtype=np.float32)

        logger.info(f"📊 Processed Input Data: {input_data}")

        # Make Prediction
        prediction = model.predict(input_data)[0]
        prediction_rounded = round(float(prediction), 2)

        logger.info(f"💰 Predicted Insurance Charges: ₹{prediction_rounded:.2f}")

        return {"predicted_charges": f"₹{prediction_rounded:.2f}"}

    except ValueError as ve:
        logger.error(f"⚠️ Validation Error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"❌ Error in prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Custom 404 Error Handler
@app.exception_handler(404)
def not_found_error(request, exc):
    return {"error": "❌ Endpoint not found. Check the URL and try again."}

# Run the Application
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
