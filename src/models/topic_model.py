import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import joblib

def fit_lda_model(docs, n_topics=5, max_iter=10, random_state=37):
    """
    Fit LDA topic model on a list of documents.
    
    Args:
        docs (list of str): List of cleaned documents (text).
        n_topics (int): Number of topics to extract.
        max_iter (int): Number of iterations.
        random_state (int): Random seed.

    Returns:
        model (LatentDirichletAllocation): Fitted LDA model.
        vectorizer (CountVectorizer): Fitted CountVectorizer.
        topic_word_df (pd.DataFrame): Top words per topic.
    """
    # Vectorize documents with CountVectorizer (bag-of-words)
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(docs)

    # Fit LDA model
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=max_iter,
                                    random_state=random_state)
    lda.fit(doc_term_matrix)

    # Extract top words per topic
    words = vectorizer.get_feature_names_out()
    topic_words = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-11:-1]
        top_words = [words[i] for i in top_words_idx]
        topic_words[topic_idx] = top_words
    
    topic_word_df = pd.DataFrame.from_dict(topic_words, orient='index',
                                           columns=[f'word_{i+1}' for i in range(10)])
    return lda, vectorizer, topic_word_df

def assign_topics(lda, vectorizer, docs):
    """
    Assign dominant topic for each document.

    Args:
        lda: fitted LDA model
        vectorizer: fitted CountVectorizer
        docs: list of cleaned documents
    
    Returns:
        list of dominant topic indices for each doc
    """
    doc_term_matrix = vectorizer.transform(docs)
    topic_distributions = lda.transform(doc_term_matrix)
    dominant_topics = topic_distributions.argmax(axis=1)
    return dominant_topics.tolist()

if __name__ == "__main__":
    sample_docs = [
    "The product quality is excellent and delivery was fast.",
    "Customer service was rude and unhelpful.",
    "I love the easy-to-use interface and intuitive design.",
    "The price is too high compared to competitors.",
    "Shipping was delayed and packaging was damaged.",
    "The website navigation is confusing and slow.",
    "The staff was very friendly and resolved my issue quickly.",
    "I am disappointed with the battery life of this device.",
    "Great value for the price paid.",
    "The refund process took longer than expected.",
    "I am satisfied with the overall experience.",
    "The app crashes frequently and needs improvement.",
    "Excellent communication throughout the purchase.",
    "Product did not meet my expectations based on description.",
    "Fast response time from the support team.",
    "I would recommend this to my friends and family.",
    "The checkout process was complicated and frustrating.",
    "I received a defective item and had to return it.",
    "High-quality materials used in the product.",
    "Customer support was not helpful at all.",
    "The delivery person was polite and professional.",
    "I appreciate the frequent updates and notifications.",
    "The color options are limited but the style is nice.",
    "Overall, a pleasant shopping experience.",
    "The product packaging was eco-friendly and neat."
    ]

    print("Fitting LDA model documents")
    lda_model, vectorizer, topics_df = fit_lda_model(sample_docs, n_topics=3)

    print("Top words per topic:")
    print(topics_df)

    dominant_topics = assign_topics(lda_model, vectorizer, sample_docs)
    print("Dominant topic for each document:")
    for i, topic_idx in enumerate(dominant_topics):
        print(f"Doc {i}: Topic {topic_idx}")

