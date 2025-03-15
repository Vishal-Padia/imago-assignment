import pytest
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from src.model_training import train_model, load_data

# Sample test data
TEST_DATA = pd.DataFrame(
    {
        "0": [0.1, 0.2, 0.3, 0.4, 0.5],
        "1": [0.5, 0.6, 0.7, 0.8, 0.9],
        "2": [0.2, 0.3, 0.4, 0.5, 0.6],
        "vomitoxin_ppb": [1.1, 2.2, 3.3, 4.4, 5.5],
    }
)


# Test: Data loading
def test_load_data(tmp_path):
    test_file = tmp_path / "test_data.csv"
    TEST_DATA.to_csv(test_file, index=False)

    data = load_data(str(test_file))

    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert "vomitoxin_ppb" in data.columns


# Test: Model training process
def test_train_model(tmp_path):
    test_file = tmp_path / "test_data.csv"
    TEST_DATA.to_csv(test_file, index=False)

    model_path = tmp_path / "xgboost_model.pkl"

    # Train model
    train_model(str(test_file))

    # Ensure model is saved
    assert model_path.exists()

    # Load and test model
    model = joblib.load(str(model_path))
    X = TEST_DATA.drop(["vomitoxin_ppb"], axis=1)
    preds = model.predict(X)

    assert len(preds) == len(X)
    assert all(isinstance(val, float) for val in preds)


# Test: Ensure hyperparameter tuning runs without errors
@pytest.mark.parametrize("trials", [1, 5])  # Using low trials for speed
def test_optuna_hyperparameter_optimization(monkeypatch, trials):
    from src.model_training import objective

    # Mock data for faster Optuna testing
    X_mock, _, y_mock, _ = train_test_split(
        TEST_DATA.drop("vomitoxin_ppb", axis=1),
        TEST_DATA["vomitoxin_ppb"],
        test_size=0.2,
        random_state=42,
    )

    global X_train, y_train, X_val, y_val
    X_train, y_train, X_val, y_val = X_mock, y_mock, X_mock, y_mock

    # Mock Optuna trial
    class MockTrial:
        def suggest_int(self, name, low, high, step=1):
            return low

        def suggest_float(self, name, low, high):
            return low

    # Ensure no exceptions are raised during the objective function
    try:
        objective(MockTrial())
    except Exception as e:
        pytest.fail(f"Optuna optimization failed: {e}")
