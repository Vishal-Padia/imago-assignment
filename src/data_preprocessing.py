import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path):
    """Load dataset and inspect basic details."""
    data = pd.read_csv(file_path)
    print("Data Info:\n", data.info())
    print("Summary Statistics:\n", data.describe())
    return data


def handle_missing_values(data):
    """Handle missing values by imputing with the mean."""
    imputer = SimpleImputer(strategy="mean")
    first_column = data.iloc[:, 0]
    data_to_impute = data.iloc[:, 1:]
    imputed_data = pd.DataFrame(
        imputer.fit_transform(data_to_impute), columns=data_to_impute.columns
    )
    return pd.concat([first_column, imputed_data], axis=1)


def normalize_data(data):
    """Normalize the dataset using StandardScaler."""
    scaler = StandardScaler()
    first_column = data.iloc[:, 0]
    data_to_transform = data.iloc[:, 1:]
    standard_data = scaler.fit_transform(data_to_transform)
    return pd.concat(
        [first_column, pd.DataFrame(standard_data, columns=data.columns[1:])], axis=1
    )


def visualize_data(data):
    """Generate data visualizations."""
    # data_to_visualize = data.iloc[:, 1:]

    # plt.figure(figsize=(12, 6))
    # sns.boxplot(data=data_to_visualize)
    # plt.title("Boxplot for Outlier Analysis")
    # plt.show()

    # plt.figure(figsize=(12, 6))
    # sns.heatmap(data_to_visualize.corr(), cmap="coolwarm", annot=True)
    # plt.title("Correlation Heatmap")
    # plt.show()


def preprocess_data(file_path):
    data = load_data(file_path)
    data = handle_missing_values(data)
    # visualize_data(data)
    data = normalize_data(data)
    return data


if __name__ == "__main__":
    file_path = "data/data.csv"
    processed_data = preprocess_data(file_path)
    processed_data.to_csv("data/processed_data.csv", index=False)
    print("Preprocessing complete. Data saved to 'data/processed_data.csv'")
