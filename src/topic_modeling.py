# sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import AgglomerativeClustering, OPTICS, KMeans

# clustering
import fasttext
import hdbscan
import umap.umap_ as umap
import torch
from transformers import BertTokenizer, BertModel, FeatureExtractionPipeline

# data
import numpy as np
import pandas as pd

# misc
import warnings
warnings.filterwarnings('ignore')
import _pickle
from collections import Counter
import time


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


def bert_preprocessing(data):
    """
    Transform incoming data for using BERT framework
    - param data: comments in a list
    """
    print("Initializing Bert tokenizer ...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    all_tensors = list()
    for text in data:
        # Encode the text to ids representing the bert tokens.
        ids = tokenizer.encode(text)
        tokens = tokenizer.convert_ids_to_tokens(ids)

        # Convert to torch tensor.
        tensor = torch.tensor([ids])
        all_tensors.append(tensor)
    # all_tensors: representation of each input comment
    #              in an embedding space
    return all_tensors


def get_bert_embeddings(torch_data):
    """
    Get embeddings from Bert English model.
    - param torch_data: Preprocessed data from bert_preprocessing
    """
    print("Initializing Bert model ...")
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    # set bert model to eval mode
    model.eval()

    all_embeddings = list()
    n_tensors = len(torch_data)
    with torch.no_grad():
        # little time measurement
        start = time.process_time()
        for i, data_tensor in enumerate(torch_data):
            out = model(input_ids=data_tensor)
            # this whole thing calculates the concatenated embeddings
            # from the last for bert model layers
            hidden_states = out[2]
            last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
            cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
            all_embeddings.append([emb for emb in cat_hidden_states.numpy().squeeze()])
            # track embedding process
            if i % 1000 == 0 and i != 0:
                print(f"Embedded {i} of {n_tensors} sentences...")
    end = time.process_time()
    embedding_time = end-start
    print(f"Time needed for embedding of {n_tensors} samples: {embedding_time}")

    return all_embeddings


def reduce_dimensions(clustering_data, metric, n_neighbors):
    """
    Dimensionality reduction for improved clustering.
    - param clustering_data: input data frame with preprocessed text and encodings
    - param metric: Dimensions of the output embeddings. Our experience is that dimension of 3 performs well.
    - param n_neighbors: Number of neighbors
    """
    # PCA dimensionality reduction to improve performance and maintain variance.
    clustering_data = PCA(n_components=150).fit_transform(clustering_data)
    # UMAP dimensionality reduction to more cleanly separate clusters and improve performance.
    print('Performing dim reduction ...')
    reducer = umap.UMAP(metric=metric, random_state=42, min_dist=0.0, spread=5, n_neighbors=n_neighbors)
    clustering_data = reducer.fit(clustering_data).embedding_

    return clustering_data


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
        n_clusters = 17
        kmean = KMeans(n_clusters=n_clusters,
                           max_iter=100,
                           init="k-means++",
                           n_init=1)

        classes = kmean.fit_predict(clustering_data)
        return classes


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
    data = data[data['comment_clean'].map(lambda x: len(x) > 0)]

    try:
    # Load embeddings if already calculated.
        print('Loading embeddings ...')
        if embeddings == "fasttext":
            embedding_file = "_mean_embeddings"
        elif embeddings == "bert":
            embedding_file = "_mean_embeddings"

        with open(embedding_file, mode="rb") as fp:
            # Export to file.
            data['embedding'] = _pickle.load(fp)

    except FileNotFoundError as e:
        # Compute embeddings if not calculated.
        print(f'Getting embeddings: {embeddings} ...\n')
        if embeddings == 'fasttext':
            data['embeddings'] = get_fasttext_embeddings(data['comment_clean'])

        elif embeddings == 'word2vec':
            data['embeddings'] = get_word2vec_embeddings(data['comment_clean'])

        elif embeddings == 'bert':
            # Preprocessing for bert and torch models
            torch_data = bert_preprocessing(data['comment_raw'])
            data['embeddings'] = get_bert_embeddings(torch_data)

            print("\nSave BERT embeddings...")
            with open("./topic_modeling_embeddings/bert_embeddings.pickle", mode="wb") as file_handle:
                _pickle.dump(data['embeddings'], file_handle)
            print("Exit code!")
            exit()
        else:
            print('Selected embeddings not supported.')
            exit()

        # Get mean embeddings.
        print('Computing weighted mean embeddings ...')
        # Compute word frequency for weighted sentence vectors.
        word_frequency = get_word_frequency(data['comment_clean'])
        # Compute sentence embeddings as weighted average of tokens.
        data['embeddings'] = get_weighted_sentence_vectors(data['embeddings'], data['comment_clean'], word_frequency)
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
        clustering_data = reduce_dimensions(clustering_data, 'cosine', 19)
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
