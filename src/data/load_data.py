import pandas as pd
import os

def load_reviews(path: str = "data/raw/Reviews.csv", text_column: str = "Text"):
    """
    Loads customer reviews from CSV file.

    Args:
        path (str): Path to the CSV file
        text_column (str): column name for the review text.
    
    Returns:
        pd.DataFrame: DataFrame containing the review text. 
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    dataframe = pd.read_csv(path)

    if text_column not in dataframe.columns:
        raise ValueError("The following Column was not found in the file:", text_column)
    
    dataframe = dataframe.dropna(subset=[text_column])
    dataframe = dataframe.reset_index(drop=True)

    print(f"[INFO] Loaded {len(dataframe)} reviews from {path}")
    return dataframe

#Test load data function with default Reviews.csv
if __name__ == "__main__":
    dataframe = load_reviews()
    print(dataframe.head())
