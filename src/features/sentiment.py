import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Literal, cast

vader = SentimentIntensityAnalyzer()

def analyze_sentiment(
    texts: pd.Series,
    method: Literal["vader", "textblob", "hybrid"] = "vader"
) -> pd.DataFrame:
    """
    Analyze sentiment using VADER, TextBlob, or Average of both.

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
        elif method == "hybrid":
            scores = vader.polarity_scores(text)
            vader_compound = scores["compound"]

            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity  # type: ignore

            # Average polarity
            avg_score = (vader_compound + textblob_polarity) / 2

            result = {
                "vader_compound": vader_compound,
                "textblob_polarity": textblob_polarity,
                "avg_score": avg_score,
                "label": _label_from_compound(avg_score)
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

if __name__ == "__main__":
    sample_reviews = pd.Series([
        "I absolutely love this product! It exceeded all my expectations.",
        "Terrible experience, would not recommend to anyone.",
        "It was okay, neither great nor terrible.",
        "Fantastic service and friendly staff!",
        "The product quality is bad and the delivery was slow."
    ])

    for method in ["vader", "textblob", "hybrid"]:
        method_literal = cast(Literal["vader", "textblob", "hybrid"], method)
        print(f"\nTesting sentiment analysis: {method}")
        df = analyze_sentiment(sample_reviews, method=method_literal)
        print(df)
