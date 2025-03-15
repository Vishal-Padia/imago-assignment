import pytest
import pandas as pd
import numpy as np
from src.prediction import predict, load_model, preprocess_input

# Sample test data
TEST_DATA = pd.DataFrame(
    {"0": [0.1, 0.2, 0.3], "1": [0.5, 0.6, 0.7], "2": [0.2, 0.3, 0.4]}
)

# Sample JSON data for testing
VALID_JSON_INPUT = {"0": 0.15, "1": 0.55, "2": 0.25}

INVALID_JSON_INPUT = {"0": "invalid_value", "1": 0.55, "2": 0.25}


# Test: Model loading
def test_load_model():
    model_path = "../models/xgboost_model.pkl"
    model = load_model(model_path)
    assert model is not None


# Test: Prediction with valid data
def test_predict_with_valid_data():
    model_path = "../models/xgboost_model.pkl"
    model = load_model(model_path)

    preds = predict(model, TEST_DATA)
    assert len(preds) == len(TEST_DATA)
    assert all(isinstance(val, float) for val in preds)


# Test: Prediction with valid JSON data
def test_predict_with_valid_json():
    model_path = "../models/xgboost_model.pkl"
    model = load_model(model_path)

    json_data = pd.DataFrame([VALID_JSON_INPUT])
    preds = predict(model, json_data)

    assert len(preds) == 1
    assert isinstance(preds[0], float)


# Test: Prediction with invalid JSON data
def test_predict_with_invalid_json():
    model_path = "../models/xgboost_model.pkl"
    model = load_model(model_path)

    with pytest.raises(Exception):
        invalid_data = pd.DataFrame([INVALID_JSON_INPUT])
        predict(model, invalid_data)


# Test: Data preprocessing - Ensure scaling handles NaNs correctly
def test_preprocess_input():
    input_data = TEST_DATA.copy()
    input_data.iloc[1, 1] = np.nan  # Inject missing value

    processed_data = preprocess_input(input_data)
    assert processed_data.isnull().sum().sum() == 0  # Ensure no NaNs remain
