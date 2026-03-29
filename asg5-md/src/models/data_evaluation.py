import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

while not os.path.exists(os.path.join(CURRENT_DIR, "pyproject.toml")):
    parent = os.path.dirname(CURRENT_DIR)
    if parent == CURRENT_DIR:
        raise FileNotFoundError("pyproject.toml not found in any parent directory")
    CURRENT_DIR = parent

# Paths
data_path = os.path.join(CURRENT_DIR, "src", "data", "raw", "train.csv")  # or test.csv
pipeline_path = os.path.join(CURRENT_DIR, "artifacts", "pipeline.pkl")


if not os.path.exists(data_path):
    raise FileNotFoundError(f"{data_path} not found")

df = pd.read_csv(data_path)

# Drop columns not used in the pipeline
df = df.drop(["Name", "PassengerId"], axis=1, errors="ignore")

if "Transported" not in df.columns:
    raise KeyError("Transported column not found in dataset")

y_test = df["Transported"]
X_test = df.drop("Transported", axis=1)

#loading pipeline
if not os.path.exists(pipeline_path):
    raise FileNotFoundError(f"{pipeline_path} not found. Train the pipeline first.")

pipeline = joblib.load(pipeline_path)

#evaluation
y_pred = pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n=== Model Evaluation ===")
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

