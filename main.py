import pandas as pd
from sentence_transformers import SentenceTransformer, util
import umap.umap_ as umap
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import json


def load_data(file_path='./arxiv-metadata-oai-snapshot.json'):
    # Load the data to DataFrame
    # Initialize an empty list to hold the relevant data entries
    relevant_data = []
    c = 0
    # Open the file and read it line by line
    with open(file_path, 'r') as f:
        for line in f:
            c += 1
            data = json.loads(line)
            relevant_data.append(data)
            # to test the code on small data
            if c == 1000:
                break
    # Convert the relevant_data list to a DataFrame
    return pd.DataFrame(relevant_data)


def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)
    return tf_idf, count

def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names_out()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words


def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                   .abstract
                   .count()
                   .reset_index()
                   .rename({"Topic": "Topic", "abstract": "Size"}, axis='columns')
                   .sort_values("Size", ascending=False))
    return topic_sizes


if __name__ == '__main__':

    # Load the data
    df = load_data(file_path = '/Users/steveabecassis/Desktop/ProjectDataMining/arxiv-metadata-oai-snapshot.json')

    # Extract embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings_abstracts = model.encode(df['abstract'],show_progress_bar=True)

    # Reduce dimensionality
    umap_model = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine',verbose=True)
    reduce_embeddings = umap_model.fit_transform(embeddings_abstracts)

    # Cluster reduced embeddings
    hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom',prediction_data=True)
    hdbscan_model.fit_predict(embeddings_abstracts)
    # clusters = np.unique(hdbscan_model.labels_,return_counts=True)
    print(f"There is {len(np.unique(hdbscan_model.labels_))} topics")
    df['Topic'] = hdbscan_model.labels_

    # Tokenize topics
    vectorizer_model = CountVectorizer(stop_words="english")
    cv = vectorizer_model.fit_transform(df['abstract'])
    vectorizer_model.get_feature_names_out()

    # Extract topic words
    docs_per_topic = df.groupby(['Topic'], as_index=False).agg({'abstract': ' '.join})
    tf_idf, count = c_tf_idf(docs_per_topic.abstract.values, m=len(df))
    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
    topic_sizes = extract_topic_sizes(df)

    topic_sizes.head(10)

    # Top 10 words of cluster 1
    pd.DataFrame(top_n_words[1][:10])


    # Visualisation






