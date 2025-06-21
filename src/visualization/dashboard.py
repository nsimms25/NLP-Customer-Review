import streamlit as st
import pandas as pd

from src.data.load_data import load_reviews
from src.data.clean_text import clean_texts
from src.features.vectorizer import vectorize_text
from src.features.sentiment import analyze_sentiment
from src.models.topic_model import fit_lda_model, assign_topics

st.set_page_config(layout="wide")
st.title("Customer Insights Dashboard")

# Load data
uploaded_file = st.file_uploader("Upload a CSV of customer reviews", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {len(df)} reviews")
else:
    df = load_reviews("data/raw/sub_sample_50k.csv")
    st.info("Using sample data from `sub_sample_50k.csv`")

# Preprocess
df = df[df["Summary"].notna()]  # Drop rows with NaN Summary
df["Summary"] = df["Summary"].astype(str)
df["cleaned"] = clean_texts(df["Summary"].tolist())

# Filter out empty or invalid cleaned texts
valid_mask = df["cleaned"].apply(lambda x: isinstance(x, str) and x.strip() != "")
filtered_df = df.loc[valid_mask].copy()
filtered_texts = filtered_df["cleaned"].tolist()

st.write(f"Number of valid cleaned texts: {len(filtered_texts)}")

# Sentiment analysis
st.header("Sentiment Analysis")
sentiment_df = analyze_sentiment(filtered_df["cleaned"], method="hybrid")
filtered_df = pd.concat([filtered_df.reset_index(drop=True), sentiment_df.reset_index(drop=True)], axis=1)
st.dataframe(filtered_df[["Summary", "cleaned", "label"]].sample(10))

sentiment_counts = filtered_df["label"].value_counts()
st.bar_chart(sentiment_counts)

# Topic modeling
st.header("Topic Modeling")
dtm, vectorizer = vectorize_text(filtered_texts)
lda_model, vectorizer, topic_word_df = fit_lda_model(dtm, vectorizer, n_topics=5)

st.subheader("Top Words Per Topic")
st.dataframe(topic_word_df)

doc_topics = assign_topics(lda_model, vectorizer, filtered_texts)
filtered_df["dominant_topic"] = doc_topics

st.subheader("Topic Distribution")
st.bar_chart(filtered_df["dominant_topic"].value_counts())

# Filtered view by topic
st.subheader("🔎 Explore Reviews")
topic_filter = st.selectbox("Filter by Topic", options=sorted(filtered_df["dominant_topic"].unique()))
filtered_view = filtered_df[filtered_df["dominant_topic"] == topic_filter]
st.dataframe(filtered_view[["Summary", "label", "dominant_topic"]].head(10))
