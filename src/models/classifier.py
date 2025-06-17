import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
import os

def train_classifier(X, y, model_type="logistic", save_path=None):
    """
    Train a text classifier.
    
    Args:
        X: Feature matrix (vectorized text)
        y: Labels (e.g., sentiment, risk)
        model_type: 'logistic' or 'rf'
        save_path: Path to save the model (optional)
    
    Returns:
        Trained model, test set, and predictions
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37)

    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "rf":
        model = RandomForestClassifier()
    else:
        raise ValueError("Unsupported model type")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report: ")
    print(classification_report(y_test, y_pred))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        dump(model, save_path)
        print(f"Model saved to {save_path}")

    return model, (X_test, y_test), y_pred

if __name__ == "__main__":
    from src.data.load_data import load_reviews
    from src.data.clean_text import clean_text
    from src.features.vectorizer import vectorize_text

    print("Loading and preparing data...")
    df = load_reviews("data/raw/sub_sample.csv")
    print(df.head())
    df['clean'] = df['Summary'].apply(clean_text)

    df['label'] = df['Summary'].str.contains("good|great|excellent|love", case=False).astype(int)

    X, feature_names = vectorize_text(df['clean'], method='tfidf')
    y = df['label']

    print("Training classifier...")
    model, test_data, preds = train_classifier(X, y, model_type="logistic", save_path="outputs/models/classifier.joblib")

"""
OUTPUT:

Training classifier...
Confusion Matrix:
[[147   0]
 [ 12  41]]
Classification Report: 
              precision    recall  f1-score   support

           0       0.92      1.00      0.96       147
           1       1.00      0.77      0.87        53

    accuracy                           0.94       200
   macro avg       0.96      0.89      0.92       200
weighted avg       0.94      0.94      0.94       200

Model saved to outputs/models/classifier.joblib
"""