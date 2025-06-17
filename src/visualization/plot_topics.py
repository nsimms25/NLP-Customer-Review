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



