import os
import joblib
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


# Load processed data
def load_data(file_path):
    return pd.read_csv(file_path)


# Objective function for Optuna
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
    }

    model = xgb.XGBRegressor(**params)

    # using KFold cross-validation with 5 splits
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list = []
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        model.fit(X_train_fold, y_train_fold)
        preds = model.predict(X_val_fold)
        rmse_list.append(root_mean_squared_error(y_val_fold, preds))

    return sum(rmse_list) / len(rmse_list)


# Model training function
def train_model(data_path):
    data = load_data(data_path)
    X = data.drop(["hsi_id", "vomitoxin_ppb"], axis=1)  # Features
    y = data["vomitoxin_ppb"]  # Target

    global X_train, X_val, y_train, y_val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print(f"Best Parameters: {study.best_params}")

    # Final model training with best params
    best_model = xgb.XGBRegressor(**study.best_params)
    best_model.fit(X_train, y_train)

    # Save model
    joblib.dump(best_model, "models/xgboost_model.pkl")
    print("Model training complete. Model saved to 'models/xgboost_model.pkl'")

    # Evaluation
    evaluate_model(best_model, X_val, y_val)


# Evaluation function
def evaluate_model(model, X, y):
    preds = model.predict(X)
    print(f"MAE: {mean_absolute_error(y, preds):.4f}")
    print(f"RMSE: {root_mean_squared_error(y, preds):.4f}")
    print(f"R² Score: {r2_score(y, preds):.4f}")


if __name__ == "__main__":
    train_model("data/processed_data.csv")
