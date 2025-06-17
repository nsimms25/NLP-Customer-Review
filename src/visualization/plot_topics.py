import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyLDAvis
import pyLDAvis.lda_model

def plot_topic_distributions(doc_topics, save_path=None):
    """
    Plots how many documents are assigned to each topic.
    
    Parameters:
        doc_topics (List[int] or Series): Dominant topic for each document
        save_path (str): Path to save the plot image (optional)
    """
    topic_counts = pd.Series(doc_topics).value_counts().sort_index()
    sns.barplot(x=topic_counts.index, y=topic_counts.values, palette="Blues_d")
    plt.title("Document Count per Topic")
    plt.xlabel("Topic")
    plt.ylabel("Number of Documents")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Bar chart saved to {save_path}")
    else:
        plt.show()

def generate_interactive_lda_plot(lda_model, dtm, vectorizer, output_html="lda_vis.html"):
    """
    Generates an interactive pyLDAvis HTML visualization.
    
    Parameters:
        lda_model: Trained LDA model from scikit-learn
        dtm: Document-term matrix (used during LDA training)
        vectorizer: Fitted CountVectorizer or TfidfVectorizer
        output_html (str): File path to save the HTML visualization
    """
    print("Generating interactive LDA visualization...")
    vis_data = pyLDAvis.lda_model.prepare(lda_model, dtm, vectorizer)
    pyLDAvis.save_html(vis_data, output_html)
    print(f"Interactive LDA visualization saved to: {output_html}")

if __name__ == "__main__":
    import pickle
    from sklearn.decomposition import LatentDirichletAllocation
    from src.data.load_data import load_reviews
    from src.data.clean_text import clean_texts
    from src.features.vectorizer import vectorize_text
    from src.models.topic_model import fit_lda_model, assign_topics

    # Load and preprocess
    df = load_reviews("data/raw/sub_sample_100k.csv")
    texts = clean_texts(df["Summary"].tolist())
    dtm, vectorizer = vectorize_text(texts)
    lda_model = fit_lda_model(dtm, n_topics=3)

    # Get dominant topics
    doc_topics = assign_topics(lda_model, dtm, docs=texts)
    plot_topic_distributions(doc_topics, save_path="outputs/figures/topic_distribution.png")

    # Generate interactive visualization
    generate_interactive_lda_plot(lda_model, dtm, vectorizer, output_html="outputs/figures/lda_vis.html")
