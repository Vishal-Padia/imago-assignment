import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


# Load processed data and trained model
def load_data(file_path):
    return pd.read_csv(file_path)


def load_model(model_path):
    return joblib.load(model_path)


# Evaluation metrics
def evaluate_model(model, X, y):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = root_mean_squared_error(y, preds)
    r2 = r2_score(y, preds)

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Scatter plot: Actual vs Predicted
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y, y=preds, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--")
    plt.title("Actual vs Predicted")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid(True)
    plt.show()

    # Residual Plot
    plt.figure(figsize=(8, 6))
    sns.residplot(x=y, y=preds, lowess=True, line_kws={"color": "red"})
    plt.title("Residual Plot")
    plt.xlabel("Actual")
    plt.ylabel("Residuals")
    plt.grid(True)
    plt.show()


# Feature importance can't be calculated without the feature names

if __name__ == "__main__":
    data_path = "data/processed_data.csv"
    model_path = "models/xgboost_model.pkl"

    data = load_data(data_path)
    model = load_model(model_path)

    X = data.drop(["hsi_id", "vomitoxin_ppb"], axis=1)
    y = data["vomitoxin_ppb"]

    evaluate_model(model, X, y)
