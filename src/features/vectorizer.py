import pandas as pd
import numpy as np
from typing import Tuple, Union, List
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import issparse, spmatrix

def vectorize_text(
    texts: List[str],
    method: str = "tfidf",
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 1),
    min_df: int = 2,
    max_df: float = 0.95
) -> Tuple[pd.DataFrame, Union[CountVectorizer, TfidfVectorizer]]:
    """
    Vectorizes text data using TF-IDF or Bag-of-Words.
    Returns DataFrame and fitted vectorizer.
    """
    vectorizer: Union[CountVectorizer, TfidfVectorizer]
    if method == "bow":
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df
        )
    elif method == "tfidf":
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df
        )
    else:
        raise ValueError("method must be 'bow' or 'tfidf'")

    matrix: Union[spmatrix, np.ndarray] = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # Always convert to ndarray for DataFrame creation
    data: np.ndarray = matrix.toarray() if issparse(matrix) else np.asarray(matrix) # type: ignore

    features_df = pd.DataFrame(data, columns=feature_names)
    return features_df, vectorizer

if __name__ == "__main__":
    sample_texts = pd.Series([
        "The product was excellent and well made.",
        "Terrible customer service. Will not buy again.",
        "Great value for the price!",
        "The delivery was delayed but the product quality was good.",
        "Awful experience. Broken item and no refund offered."
    ])

    # Run the vectorizer function
    features_df, vectorizer = vectorize_text(
        texts=sample_texts.tolist(),
        method="tfidf",
        max_features=10,
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=1,
        max_df=1.0
    )

    # Print results
    print("Vectorized Feature Names:")
    print(vectorizer.get_feature_names_out())

    print("\nFeature DataFrame (shape = {}):".format(features_df.shape))
    print(features_df.head())

