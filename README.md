# NLP-Driven Customer Experience Insights

This project uses natural language processing (NLP) to extract risk and experience signals from unstructured customer support data. It demonstrates end-to-end capabilities from data ingestion to model deployment and visualization.

## Features
- Text cleaning and preprocessing pipeline
- Sentiment analysis (rule-based and ML)
- Topic modeling (LDA, NMF)
- Predictive churn modeling (optional)
- Interactive visualization dashboard
- Model documentation for governance

## Stack

- Python, scikit-learn, spaCy, NLTK
- gensim, pyLDAvis, matplotlib/seaborn
- Streamlit / FastAPI for deployment
- Docker (optional), MLflow (optional)

## Folder Structure

See [`project structure`](#) below for details.

## Example Output

- Top 5 topics from negative reviews
- Sentiment trend over time
- Risk-prone themes that correlate with escalations

## Data Source
Raw data from https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews?resource=download

## Project Structure
<pre>
nlp-customer-insights/
├── data/
│   ├── raw/                     # Unprocessed text files (e.g., tickets, reviews)
│   ├── interim/                 # Cleaned/preprocessed data
│   └── processed/               # Tokenized, vectorized, and final datasets
│
├── src/
│   ├── data/
│   │   ├── load_data.py         # Load and sample text data
│   │   └── clean_text.py        # Preprocessing: stopwords, stemming, etc.
│   │
│   ├── features/
│   │   ├── vectorizer.py        # TF-IDF, word embeddings
│   │   └── sentiment.py         # Rule-based or ML sentiment scoring
│   │
│   ├── models/
│   │   ├── topic_model.py       # LDA, NMF models
│   │   └── classifier.py        # Optional churn/risk prediction model
│   │
│   └── visualization/
│       ├── plot_topics.py       # pyLDAvis, seaborn/t-SNE
│       └── dashboard.py         # Streamlit or Plotly Dash UI
│
├── outputs/
│   ├── figures/                 # Visualizations
│   └── reports/
│       └── model_card.md        # MRM-style documentation (explainability, fairness)
│
├── app/
│   └── main.py                  # FastAPI app
│
├── requirements.txt             # Dependencies
├── README.md                    # Project overview
└── setup.py                     # Optional Python packaging script
</pre>
