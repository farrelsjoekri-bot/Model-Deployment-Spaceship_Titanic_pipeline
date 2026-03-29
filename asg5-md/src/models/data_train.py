import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from pipelines.data_pipeline import build_pipeline
# Get absolute path of current file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up until we find project root (where pyproject.toml exists)
while not os.path.exists(os.path.join(CURRENT_DIR, "pyproject.toml")):
    CURRENT_DIR = os.path.dirname(CURRENT_DIR)

# Build path to data
data_path = os.path.join(CURRENT_DIR,"src", "data", "raw", "train.csv")

print("Reading from:", data_path)  # debug

def train_model():
    df = pd.read_csv(data_path)

    df = df.drop(["Name", "PassengerId"], axis=1)
    y = df["Transported"]
    X = df.drop("Transported", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(X_train)

    pipeline.fit(X_train, y_train)

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(pipeline, "artifacts/pipeline.pkl")

    print("Pipeline has been trained and saved.", flush=True)

    return X_test, y_test


if __name__ == "__main__":
    train_model()