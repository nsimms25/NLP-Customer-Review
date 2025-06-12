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
nlp-customer-insights/
├── data/
│   ├── raw/     
│   ├── interim/ 
│   └── processed/
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── clean_text.py
│   │
│   ├── features/
│   │   ├── vectorizer.py
│   │   └── sentiment.py
│   │
│   ├── models/
│   │   ├── topic_model.py
│   │   └── classifier.py
│   │
│   └── visualization/
│       ├── plot_topics.py
│       └── dashboard.py
│
├── outputs/
│   ├── figures/
│   └── reports/
│       └── model_card.md
│
├── app/
│   └── main.py
│
├── requirements.txt
├── README.md
└── setup.py

