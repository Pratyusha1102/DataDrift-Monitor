import pandas as pd


def load_csv(file):
    """Load a CSV file and return a pandas DataFrame."""
    return pd.read_csv(file)
