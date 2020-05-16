# sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import AgglomerativeClustering, OPTICS

# clustering
import fasttext
import hdbscan
import umap.umap_ as umap

# data
import numpy as np
import pandas as pd

# misc
import warnings
warnings.filterwarnings('ignore')
import _pickle


def get_fasttext_embeddings(data):
    """
    Get embeddings from fastext English model.
    - param data: pd dataframe
    """
    # Load FastText common crawl model.
    model = fasttext.load_model('crawl-300d-2M-subword.bin')
    # Get embeddings for each word for each comment.
    return data['comment'].apply(lambda comment: [model[tok] for tok in comment] if len(comment) > 0 else [])


def get_word2vec_embeddings(data):
    """
    Get embeddings from word2vec GoogleNews model.
    - param data: pd dataframe
    """
    pass


def get_bert_embeddings(data):
    """
    Get embeddings from Bert English model.
    - param data: pd dataframe
    """
    pass


def get_cluster_ids(clustering_data, cluster_algorithm):
    """
    Run clustering algorithm on comment mean embeddings.
    - param clustering_data: list of np arrays (mean word embeddings)
    - param cluster_algorithm: str (type of cluster algorithm)
    """
    print(f'Running clustering algorithm: {cluster_algorithm} ...')
    if cluster_algorithm == 'hdbscan':
        # Instantiate the hdbscan clusterer.
        clusterer = hdbscan.HDBSCAN(algorithm='best',
                            alpha=1.0,
                            approx_min_span_tree=True,
                            gen_min_span_tree=False,
                            leaf_size=40,
                            metric='euclidean',
                            min_cluster_size=100,
                            min_samples=None,
                            p=None)

        # Fit the clusterer to the data.
        clusterer.fit(clustering_data)
        return clusterer.labels_

    elif cluster_algorithm == 'agglomerative':
        # Instantiate the agglomerative clusterer.
        clusterer = AgglomerativeClustering(n_clusters=None,
                            affinity='cosine',
                            memory=None,
                            connectivity=None,
                            compute_full_tree='auto',
                            linkage='single',
                            distance_threshold=0.55)

        # Fit the clusterer to the data.
        clusterer.fit(clustering_data)
        return clusterer.labels_

    elif cluster_algorithm == 'optics':
        # Instantiate the OPTICS clusterer.
        clusterer = OPTICS(min_samples=40,
                            metric='canberra',
                            # p=2,
                            metric_params=None,
                            cluster_method='xi',
                            eps=None,
                            xi=0.05,
                            predecessor_correction=True,
                            min_cluster_size=None,
                            algorithm='auto',
                            leaf_size=30,
                            n_jobs=None)

        clusterer.fit(clustering_data)
        return clusterer.labels_

    elif cluster_algorithm == 'kmeans':
        pass


def model_topics(data, embeddings, cluster_algorithm, normalization, dim_reduction, outliers):
    """
    Perform cluster topic modeling using comment mean embeddings.
    - param data: pd dataframe (preprocessed data)
    - param embeddings: str (type of word embeddings)
    - param cluster_algorithm (type of cluster algorithm)
    - param normalization: bool (perform normalization y/n)
    - param dim_reduction: bool (perform dimensionality reduction y/n)
    - param outliers: float (degree of expected dataset contamination)
    """
    # Drop empty values.
    data = data[data['comment'].map(lambda x: len(x) > 0)]

    try:
    # Load embeddings if already calculated.
        print('Loading embeddings ...')
        with open("_mean_embeddings", "rb") as fp:
            # Export to file.
            data['embedding'] = _pickle.load(fp)

    except FileNotFoundError as e:
        # Compute embeddings if not calculated.
        print(f'Getting embeddings: {embeddings} ...\n')
        if embeddings == 'fasttext':
            data['embeddings'] = get_fasttext_embeddings(data)

        elif embeddings == 'word2vec':
            data['embeddings'] = get_word2vec_embeddings(data)

        elif embeddings == 'bert':
            data['embeddings'] = get_bert_embeddings(data)

        else:
            print('Selected embeddings not supported.')
            exit()

        # Get mean embeddings.
        print('Computing mean embeddings ...')
        data['embeddings'] = data['embeddings'].apply(lambda x: np.mean(x, axis=0))
        # Rename column.
        data.rename(columns={'embeddings': 'embedding'}, inplace=True)
        # Store to accelerate multiple trials.
        with open("_mean_embeddings", "wb") as fp:
            _pickle.dump(data['embedding'].tolist(), fp)

    # Apply additional preprocessing.
    clustering_data = np.array(data['embedding'].tolist())
    if normalization:
        # Normalize values between 1 and 0.
        clustering_data = MinMaxScaler(feature_range=[0, 1]).fit_transform(clustering_data)
    if dim_reduction:
        # Reduce dimensions for performance while maintaining majority of variance.
        clustering_data = PCA(n_components=150).fit_transform(clustering_data)
        # UMAP dimensionality reduction to more cleanly separate clusters and improve performance.
        print('Performing dim reduction ...')
        reducer = umap.UMAP(metric='cosine', random_state=42, min_dist=0.0, spread=5, n_neighbors=19)
        clustering_data = reducer.fit(clustering_data).embedding_
        # Update the dataset with the reduced data for later visualization.
        data['PC1'] = [item[0] for item in clustering_data]
        data['PC2'] = [item[1] for item in clustering_data]

    if outliers:
        # Remove a certain percentage of outliers. Show the number of outliers.
        outlier_scores = LocalOutlierFactor(contamination=outliers).fit_predict(clustering_data)
        clustering_data = clustering_data[outlier_scores == 1]
        # Update the dataset to reflect the removed outliers.
        data = data[outlier_scores == 1]

    cluster_ids = get_cluster_ids(clustering_data, cluster_algorithm)
    # Append the cluster ids to the dataframe.
    data['cluster'] = cluster_ids
    n_clusters = data['cluster'].nunique()
    print(f'Found {n_clusters} clusters.')

    return data
