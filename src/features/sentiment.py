import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Literal

vader = SentimentIntensityAnalyzer()

def analyze_sentiment(
    texts: pd.Series,
    method: Literal["vader", "textblob"] = "vader"
) -> pd.DataFrame:
    """
    Analyze sentiment using either VADER or TextBlob.

    Args:
        texts (pd.Series): Series of cleaned review texts
        method (str): 'vader' or 'textblob'

    Returns:
        pd.DataFrame: DataFrame with sentiment scores
    """
    results = []

    for text in texts:
        if method == "vader":
            scores = vader.polarity_scores(text)
            result = {
                "compound": scores["compound"],
                "pos": scores["pos"],
                "neu": scores["neu"],
                "neg": scores["neg"],
                "label": _label_from_compound(scores["compound"])
            }
        elif method == "textblob":
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity # type: ignore
            subjectivity = blob.sentiment.subjectivity # type: ignore
            result = {
                "polarity": polarity,
                "subjectivity": subjectivity,
                "label": _label_from_compound(polarity)
            }
        else:
            raise ValueError("Method must be 'vader' or 'textblob'")

        results.append(result)

    return pd.DataFrame(results)


def _label_from_compound(score: float) -> str:
    """Convert sentiment score into a categorical label"""
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"
