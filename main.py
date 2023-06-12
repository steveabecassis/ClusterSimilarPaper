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
            if c == 1000:
                break
    # Convert the relevant_data list to a DataFrame
    return pd.DataFrame(relevant_data)


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
    clusters = np.unique(hdbscan_model.labels_)

    # Tokenize topics
    vectorizer_model = CountVectorizer(stop_words="english")






