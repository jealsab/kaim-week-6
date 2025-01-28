from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize the FastAPI app
app = FastAPI()

# Load the trained models
log_reg_model = joblib.load('notebooks/log_reg_model.pkl')
rf_model = joblib.load('notebooks/rf_model.pkl')

# Define request schema
class PredictionRequest(BaseModel):
    CurrencyCode: str
    CountryCode: str
    ProductCategory: str
    ChannelId: int
    # Add other required fields here based on your model input

# Encoding function for categorical features
def encode_features(data):
    # Example: encoding CurrencyCode, CountryCode, ProductCategory, and ChannelId
    encoding_dict = {
        'CurrencyCode': {'USD': 1, 'EUR': 2, 'GBP': 3},  # Add all relevant encodings
        'CountryCode': {'US': 1, 'UK': 2, 'FR': 3},  # Add all relevant encodings
        'ProductCategory': {
            'airtime': 1,
            'data_bundles': 2,
            'financial_services': 3,
            'movies': 4,
            'other': 5,
            'ticket': 6,
            'transport': 7,
            'tv': 8,
            'utility_bill': 9
        },
        'ChannelId': {1: 1, 2: 2, 3: 3}  # Update with actual encoding logic
    }

    # Encode each categorical feature using the encoding_dict
    for col, encoding in encoding_dict.items():
        if col in data:
            data[col] = encoding.get(data[col], 0)  # 0 for unknown categories

    return data

@app.get("/")
def home():
    return {"message": "Welcome to the Model Serving API!"}

# Define the predict endpoint for Logistic Regression model
@app.post("/predict_log_reg")
def predict_log_reg(request: PredictionRequest):
    try:
        # Convert request to DataFrame
        data = request.dict()
        input_data = pd.DataFrame([data])

        # Encode features
        input_data = input_data.apply(lambda x: encode_features(x), axis=1)

        # Make prediction
        prediction = log_reg_model.predict(input_data)
        return {"LogisticRegressionPrediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define the predict endpoint for Random Forest model
@app.post("/predict_rf")
def predict_rf(request: PredictionRequest):
    try:
        # Convert request to DataFrame
        data = request.dict()
        input_data = pd.DataFrame([data])

        # Encode features
        input_data = input_data.apply(lambda x: encode_features(x), axis=1)

        # Make prediction
        prediction = rf_model.predict(input_data)
        return {"RandomForestPrediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
