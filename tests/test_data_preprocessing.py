import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import (
    load_data,
    handle_missing_values,
    normalize_data,
)

# Sample test data for validation
TEST_DATA = pd.DataFrame(
    {
        "0": [0.1, 0.2, np.nan, 0.4, 0.5],
        "1": [0.5, 0.6, 0.7, 0.8, 0.9],
        "2": [0.2, 0.3, 0.4, np.nan, 0.6],
        "vomitoxin_ppb": [1.1, 2.2, 3.3, 4.4, 5.5],  # Target variable
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


# Test: Handling missing values
def test_handle_missing_values():
    processed_data = handle_missing_values(TEST_DATA)

    assert processed_data.isnull().sum().sum() == 0  # Ensure no missing values
    assert np.isclose(processed_data.iloc[2, 0], 0.3)  # Mean-imputed value check


# Test: Data normalization
def test_normalize_data():
    normalized_data = normalize_data(TEST_DATA.drop(["vomitoxin_ppb"], axis=1))

    assert normalized_data.shape == (5, 3)  # Same dimensions
    assert np.allclose(normalized_data.mean(), 0, atol=1e-1)  # Mean ~ 0
    assert np.allclose(normalized_data.std(), 1, atol=1e-1)  # Std ~ 1


# Test: Visualization (basic check for errors)
def test_visualize_data():
    try:
        from src.data_preprocessing import visualize_data

        visualize_data(TEST_DATA)
    except Exception as e:
        pytest.fail(f"Visualization test failed: {e}")
