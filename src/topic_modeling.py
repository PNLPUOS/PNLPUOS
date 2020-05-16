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
from collections import Counter


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


def get_weighted_sentence_vectors(sentence_vectors, sentence_tokens, word_frequency, a=1e-3):
    """
    Get weighted sentence vectors according to PCA and smooth inverse word frequency.
    Based on Aroya et al. (2016)
    - param sentence_vectors: pd Series
    - param sentence_tokens: pd Series
    - param word_frequency: dict
    """
    sentences = []
    for word_vectors, tokens in zip(sentence_vectors, sentence_tokens):
        # Create empty vector to store transformed values.
        new_vector = np.zeros(word_vectors[0].shape)
        sentence_length = len(word_vectors)
        for vec, tok in zip(word_vectors, tokens):
            # Calculate smooth inverse word frequency.
            a_value = a / (a + word_frequency[tok])
            # Adjust new vector according to product of original and frequency.
            new_vector = np.add(new_vector, np.multiply(a_value, vec))

        # Compute weighted average.
        new_vector = np.divide(new_vector, sentence_length)
        sentences.append(new_vector)

        # TODO: Use tf-idf vector for weighting.

    return sentences


def get_word_frequency(comments):
    """
    Get frequency of each word in the entire corpus.
    - param comments: pd Series
    """
    # Extract all tokens to a single list.
    vocab = list(set([word for comment in comments for word in comment]))

    word_frequency = {}
    # Calculate the frequency across the entire corpus.
    for word, count in Counter(vocab).items():
        word_frequency[word] = count / len(vocab)

    return word_frequency


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
        print('Computing weighted mean embeddings ...')
        # Compute word frequency for weighted sentence vectors.
        word_frequency = get_word_frequency(data['comment'])
        # Compute sentence embeddings as weighted average of tokens.
        data['embeddings'] = get_weighted_sentence_vectors(data['embeddings'], data['comment'], word_frequency)
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
