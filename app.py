from fastapi import FastAPI, HTTPException
import logging
import os
import pickle
from pydantic import BaseModel, Field
import numpy as np
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    with open('model/insurancemodelf.pkl', 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise RuntimeError("Model could not be loaded. Ensure 'insurancemodelf.pkl' exists and is valid.")



app = FastAPI(
    title="Medical Insurance Prediction",
    description="Predicting your max insurance coverage based on the inputs i.e BMI, age, childrens, smoker",
    version="1.0.0"
) 

class PredictionReq(BaseModel):
    BMI: int = Field(gt=0,lt=50,default=18,description="Your BMI Index")
    Age: int = Field(gt=0,description="Your Age")
    Child: int = Field(gt=-1,default=0,description="Number of childrens the person have")
    Smoker: str = Field(default="no",description="Is the person is smoker (yes/no)")

@app.get("/")
def root():
    return {"hello":"world"}
    
@app.post("/predict")
def predict(req: PredictionReq):
    try:
        logger.info(f"Recieved Request: {req}")

        smoker = 1 if req.Smoker.lower() == "yes" else 0
        input = np.array([[req.Age, req.BMI, req.Child, smoker]])

        logger.info(f"Prepared input data: {input}")

        prediction = model.predict(input)
        logger.info(f"Prediction result: {prediction}")

        return {"predicted_charges": float(prediction[0])}

    except HTTPException as e:
        raise e 
    

@app.exception_handler(404)
def not_found_error(request, exc):
    return {"error": "Endpoint not found. Please check the URL and try again."}
    
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
