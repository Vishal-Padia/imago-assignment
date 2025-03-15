import joblib
import pandas as pd


# Load the trained model
def load_model(model_path):
    return joblib.load(model_path)


# Preprocess input data
def preprocess_input(data):
    """Preprocess input data by handling missing values and scaling if needed."""

    #  Assuming preprocessed data was standardized — apply same scaling
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    # Ensure input is a DataFrame
    if isinstance(data, dict):
        data = pd.DataFrame([data])  # For JSON inputs
    elif isinstance(data, list):
        data = pd.DataFrame(data)  # For batch predictions

    # Drop irrelevant columns if present
    if "hsi_id" in data.columns:
        data = data.drop("hsi_id", axis=1)

    # Scale data
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns)


# Prediction function
def predict(model, data):
    processed_data = preprocess_input(data)
    predictions = model.predict(processed_data)
    return predictions


if __name__ == "__main__":
    model_path = "models/xgboost_model.pkl"

    # Pass in your input data here
    input_data_path = "data/sample_input.csv"

    # Load model
    model = load_model(model_path)

    # Load input data
    input_data = pd.read_csv(input_data_path)

    # Predict
    predictions = predict(model, input_data)

    # Display results
    print("Predictions:")
    for idx, pred in enumerate(predictions):
        print(f"Sample {idx + 1}: {pred:.4f}")
