import re
import string
import spacy
import pandas as pd
from typing import List, Optional
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def clean_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """
    Cleans input text by removing punctuation, numbers, stopwords, and lemmatizing.

    Args:
        text (str): The raw input text.
        remove_stopwords (bool): Whether to remove stopwords.
        lemmatize (bool): Whether to apply lemmatization.

    Returns:
        str: Cleaned text.
    """

    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip() #Extra whitespace

    if lemmatize:
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_punct]
    else:
        tokens = text.split()

    if remove_stopwords:
        tokens = [word for word in tokens if word not in STOPWORDS]

    return " ".join(tokens)

def clean_texts(texts: List[str], **kwargs) -> List[str]:
    """
    Apply clean_text to a list of strings.

    Args:
        texts (List[str]): List of raw texts.

    Returns:
        List[str]: List of cleaned texts.
    """
    cleaned = []
    for text in texts:
        if pd.isna(text) or not isinstance(text, str):
            cleaned.append("")
        else:
            cleaned.append(clean_text(text, **kwargs))
    return cleaned

if __name__ == "__main__":
    example = "This is an example of what a sentence would look like!!"
    print("[Original]:", example)
    print("[Cleaned]:", clean_text(example))
