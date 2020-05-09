# sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import silhouette_samples, silhouette_score
# from sklearn.cluster import AgglomerativeClustering

# clustering
import fasttext
import hdbscan
import umap.umap_ as umap
import torch
from transformers import BertTokenizer, BertModel, FeatureExtractionPipeline
import spacy
from spacy.pipeline import Sentencizer
import time
import datetime

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


def perform_bert_preprocessing(data: [str]):
    """
    Transform incoming data for using BERT framework
    :param data: comments in a list
    :return:
    """
    # option to split the comments to sentence level
    # with spacy sentence segmenter
    #nlp = spacy.blank("en")
    #nlp.add_pipe(nlp.create_pipe('sentencizer'))

    print("Initialize BERT Tokenizer...")
    #TODO: Implement token counter (padding or truncating)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    all_tensors = list()
    #TODO: Maybe implement the separator tokens if needed
    for text in data:
        # encode the text to ids representing the bert tokens
        ids = tokenizer.encode(text)
        tokens = tokenizer.convert_ids_to_tokens(ids)

        # convert to torch tensor
        tensor = torch.tensor([ids])
        all_tensors.append(tensor)
    # all_tensors: representation of each input comment
    #              in an embedding space
    return all_tensors

def get_bert_embeddings(data):
    """
    Get embeddings from Bert English model.
    - param data: pd dataframe
    """
    print("Initialize BERT Model...")
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    # set bert model to eval mode
    model.eval()

    all_embeddings = list()
    n_tensors = len(data)
    with torch.no_grad():
        # little time measurement
        start = time.process_time()
        for i, data_tensor in enumerate(data):
            out = model(input_ids=data_tensor)

            # this whole thing calculates the concatenated embeddings
            # from the last for bert model layers
            hidden_states = out[2]
            #sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze()

            last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
            cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
            cat_sentence_embedding = torch.mean(cat_hidden_states, dim=1).squeeze()
            all_embeddings.append(cat_sentence_embedding.numpy())
            # track embedding process
            if i % 1000 == 0 and i != 0:
                print(f"Embedded {i} of {n_tensors} sentences...")
    end = time.process_time()
    embedding_time = end-start
    print(f"Time needed for embedding of {n_tensors} samples: {embedding_time}")
    return all_embeddings

def get_representations(data_frame: pd.DataFrame, n_representations = 5):
    # get the highest rated word representations for each
    # class found by the sentence embeddings
    tfidf = TfidfVectorizer(stop_words='english')
    labels = list(set(data_frame.classes.values))

    all_representations = list()
    for label in labels:
        texts = data_frame.comments[data_frame.classes == label]
        vectorized_text = tfidf.fit_transform(texts)
        feature_array = np.array(tfidf.get_feature_names())
        tfidf_sorting = np.argsort(vectorized_text.toarray()).flatten()[::-1]
        all_representations.append(list(feature_array[tfidf_sorting][:n_representations]))

    return labels, all_representations

def get_cluster_ids(clustering_data, cluster_algorithm, n_classes: int):
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
        pass

    # basic kmeans algorithm for clustering with n classes
    elif cluster_algorithm == 'kmeans':
        n_clusters = n_classes
        kmean = KMeans(n_clusters=n_clusters,
                       max_iter=100,
                       init="k-means++",
                       n_init=1)
        classes = kmean.fit_predict(clustering_data)
        return classes


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
    # data = data[data['comment'].map(lambda x: len(x) > 0)]

    ####
    # Configurations
    n_data = 10000
    data = data[:n_data] # comments processed
    n_classes = 3 # number of classes for the clustering
    ####
    """
    make sure there is no '_mean_embedding' file so the code jumps
    to the except state and performs the bert embeddings
    """
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
            # perform the preprocessing needed for bert and torch models
            torch_data = perform_bert_preprocessing(data)
            # get the sentence embedding for all comments
            clustering_data = get_bert_embeddings(torch_data)

            #data['embeddings'] = get_bert_embeddings(data)

        else:
            print('Selected embeddings not supported.')
            exit()

        # skip the embedding preprocessing steps
        calculate_mean_embedding = False
        if calculate_mean_embedding:
            # Get mean embeddings.
            print('Computing mean embeddings ...')
            data['embeddings'] = data['embeddings'].apply(lambda x: np.mean(x, axis=0))
            # Rename column.
            data.rename(columns={'embeddings': 'embedding'}, inplace=True)
            # Store to accelerate multiple trials.
            with open("_mean_embeddings", "wb") as fp:
                _pickle.dump(data['embedding'].tolist(), fp)

    # also skip these reduction processes
    embedding_preprocessing = False
    if embedding_preprocessing:
        # Apply additional preprocessing.
        #clustering_data = np.array(data['embedding'].tolist())
        if normalization:
            # Normalize values between 1 and 0.
            clustering_data = MinMaxScaler(feature_range=[0, 1]).fit_transform(clustering_data)

        if dim_reduction:
            # Reduce dimensions for performance while maintaining majority of variance.
            clustering_data = PCA(n_components=100).fit_transform(clustering_data)

            # UMAP dimensionality reduction to more cleanly separate clusters and improve performance.
            reducer = umap.UMAP(random_state=42, min_dist=0.0, spread=5, n_neighbors=19)
            clustering_data = reducer.fit(clustering_data).embedding_
            # Update the dataset with the reduced data for later visualization.
            #data['PC1'] = [item[0] for item in clustering_data]
            #data['PC2'] = [item[1] for item in clustering_data]

        if outliers:
            # Remove a certain percentage of outliers. Show the number of outliers.
            outlier_scores = LocalOutlierFactor(contamination=outliers).fit_predict(clustering_data)
            clustering_data = clustering_data[outlier_scores == 1]
            # Update the dataset to reflect the removed outliers.
            data = data[outlier_scores == 1]

    # get the clusters from the bert embeddings by kmeans algorithm
    cluster_ids = get_cluster_ids(clustering_data, cluster_algorithm, n_classes=n_classes)
    cluster_df = pd.DataFrame({"classes": cluster_ids,
                               "comments": list(data)})
    cluster_df = cluster_df.sort_values(by="classes")
    # save the data frame to csv
    # containing: cluster label (n_classes)
    #             original comments
    cluster_df.to_csv("clustered_comments.csv", sep=";", index=None)

    # get the top five representations pre class by tfidf
    labels, representations = get_representations(cluster_df, n_representations=5)
    representations_df = pd.DataFrame({'classes':labels,
                                       'representations':[','.join(r) for r in representations]})
    # save the top n representing words per class to csv
    representations_df.to_csv("representations_per_class.csv", sep=";", index=None)

    # Append the cluster ids to the dataframe.
    #data['cluster'] = cluster_ids
    #n_clusters = data['cluster'].nunique()
    #print(f'Found {n_clusters} clusters.')

    return data
