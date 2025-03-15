# imago-assignment

## Setup Instructions  

### 1. Clone the Repository  
```bash
git clone <repository_url>
cd <repository_folder>
```

### 2. Create a Virtual Environment  
```bash
python -m venv venv
source venv/bin/activate        # For Linux/Mac
venv\Scripts\activate           # For Windows
```

### 3. Install Dependencies  
```bash
pip install -r requirements.txt
```

---

## Repository Structure  

```
├── data
│   ├── data.csv                # Original dataset
│   ├── processed_data.csv      # Processed dataset
│
├── models
│   └── xgboost_model.pkl       # Trained XGBoost model
│
├── src
│   ├── __init__.py
│   ├── main.py                 # FastAPI server
│   ├── data_preprocessing.py   # Data cleaning, normalization, visualization
│   ├── model_training.py       # XGBoost training with Optuna
│   ├── model_evaluation.py     # Model evaluation & SHAP analysis
│   ├── prediction.py           # Prediction logic
│
├── tests                       # Test scripts
│   ├── test_data_preprocessing.py
│   ├── test_model_training.py
│   ├── test_prediction.py
│
├── requirements.txt            # Required dependencies
├── README.md                   # Project documentation
```

---

## Usage Instructions  

### 1. Data Preprocessing  
To preprocess the data and generate `processed_data.csv`:  
```bash
python src/data_preprocessing.py
```

### 2. Model Training with Optuna  
To train the model and save it as `xgboost_model.pkl`:  
```bash
python src/model_training.py
```

### 3. Model Evaluation  
To evaluate the model’s performance and visualize insights:  
```bash
python src/model_evaluation.py
```

### 4. FastAPI Server  
To start the FastAPI server for predictions:  
```bash
uvicorn src.main:app --reload
```

### 5. Prediction Endpoints  

**For JSON Prediction:**  
- Endpoint: `POST /predict/json/`  
- Sample JSON Payload:  
```json
{
    "0": 0.15,
    "1": 0.20,
    "2": 0.13,
    "3": 0.27,
    "4": 0.32,
    "5": 0.19
}
```

**For CSV Prediction:**  
- Endpoint: `POST /predict/csv/`  
- Upload a CSV file (e.g., `sample_input.csv`) containing features.  

Access the Swagger documentation for interactive testing:  
 **[http://localhost:8000/docs](http://localhost:8000/docs)**  

---
## Testing

Test the core functionalities using the provided test scripts:

```bash
pytest tests/
```