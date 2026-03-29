import pandas as pd

def load_data():
    """Load data from a CSV file."""
    df = pd.read_csv("train.csv")
    print("Data loaded successfully.")
    return df
  