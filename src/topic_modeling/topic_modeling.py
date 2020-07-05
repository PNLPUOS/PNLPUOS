# sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import AgglomerativeClustering, OPTICS, KMeans
from sklearn.metrics import silhouette_score

# clustering
import fasttext
import hdbscan
import umap.umap_ as umap
import torch
from transformers import BertTokenizer, BertModel, FeatureExtractionPipeline

# data
from typing import List, Dict
import numpy as np
import pandas as pd

# misc
import warnings
warnings.filterwarnings('ignore')
import _pickle
from collections import Counter
import time
import itertools
import math

# tuning class
from topic_modeling.topic_modeling_utils.grid_search import HyperparameterTuning
# data bass class
from topic_modeling.topic_modeling_utils.mongo_accessor import MongoAccessor

MONGO = MongoAccessor()


def get_fasttext_embeddings(data: pd.DataFrame) -> pd.Series:
    """
    Get embeddings from fastext English model.
    - param data: pd dataframe
    """
    # Load FastText common crawl model.
    model = fasttext.load_model('topic_modeling\\crawl-300d-2M-subword.bin')
    # Get embeddings for each word for each comment.
    return data.apply(lambda comment: [model[tok] for tok in comment] if len(comment) > 0 else [])



def bert_preprocessing(data: List[str]) -> List[torch.tensor]:
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
    # all_tensors: representation of each input comment in an embedding space
    return all_tensors


def get_bert_embeddings(torch_data: List[torch.tensor]) -> List[np.array]:
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
            cat_sentence_embedding = torch.mean(cat_hidden_states, dim=1).squeeze()
            all_embeddings.append(cat_sentence_embedding.numpy())
            #cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
            #all_embeddings.append([emb for emb in cat_hidden_states.numpy().squeeze()])
            # track embedding process
            if i % 1000 == 0 and i != 0:
                print(f"Embedded {i} of {n_tensors} sentences...")
    end = time.process_time()
    embedding_time = end-start
    print(f"Time needed for embedding of {n_tensors} samples: {embedding_time}")

    return all_embeddings


def get_arguments(default: Dict[str, Dict], input_parameters: Dict):
    """
    update default parameters with
    input parameters
    :param default: given config
    :param input_parameters: data to over write
    :return:
    """
    for key in input_parameters:
        if key in default:
            default.update(
                {key: input_parameters[key]}
            )

    return default


def normalize_data(clustering_data: List,
                   normalization_algorithm: str="MinMaxScaler",
                   parameter_config=None ) -> List:
    """
    normalize data for later processing
    :param clustering_data:
    :param normalization_algorithm:
    :param parameter_config:
    :return:
    """
    print(f'Performing normalization {normalization_algorithm}...')

    if not parameter_config:
        parameter_config = dict()

    if normalization_algorithm == "MinMaxScaler":
        default_param = {
            'feature_range': [0, 1],
        }
        param = get_arguments(default_param, parameter_config)
        normalizer = MinMaxScaler(
            feature_range=param['feature_range']
        )

        clustering_data = normalizer.fit_transform(clustering_data)
    return clustering_data


def reduce_dimensions(clustering_data: List,
                      reduction_algorithm: str="UMAP",
                      parameter_config=None) -> List:
    """
    dimensionality reduction for improvement
    of clustering
    :param clustering_data:
    :param reduction_algorithm:
    :param parameter_config:
    :return:
    """
    # PCA dimensionality reduction to improve performance and maintain variance.
    clustering_data = PCA(n_components=150).fit_transform(clustering_data)
    # UMAP dimensionality reduction to more cleanly separate clusters and improve performance.
    print(f'Performing dim reduction {reduction_algorithm}...')

    if not parameter_config:
        parameter_config = dict()

    if reduction_algorithm == "UMAP":
        default_param = {
            'metric': 'cosine',
            'random_state': 42,
            'min_dist': 0.0,
            'spread': 4,
            'n_neighbors': 19
        }
        param = get_arguments(default_param, parameter_config)

        reducer = umap.UMAP(metric=param['metric'],
                            random_state=param['random_state'],
                            min_dist=param['min_dist'],
                            spread=param['spread'],
                            n_neighbors=param['n_neighbors'])

        clustering_data = reducer.fit(clustering_data).embedding_

    return clustering_data


def get_cluster_ids(clustering_data, cluster_algorithm="hdbscan", parameter_config=None):
    """
    Run clustering algorithm on comment mean embeddings.
    - param clustering_data: list of np arrays (mean word embeddings)
    - param cluster_algorithm: str (type of cluster algorithm)
    """
    print(f'Running clustering algorithm: {cluster_algorithm} ...')

    if not parameter_config:
        parameter_config = dict()

    if cluster_algorithm == 'hdbscan':
        default_param = {
            'algorithm': 'best',
            'alpha': 0.1,
            'approx_min_span_tree': True,
            'gen_min_span_tree': False,
            'leaf_size': 40,
            'metric': 'euclidean',
            'min_cluster_size': 100,
            'min_samples': None,
            'cluster_selection_epsilon': 0.3,
            'p': None
        }
        param = get_arguments(default_param, parameter_config)

        # Instantiate the hdbscan clusterer.
        clusterer = hdbscan.HDBSCAN(
            algorithm=param['algorithm'],
            alpha=param['alpha'],
            approx_min_span_tree=param['approx_min_span_tree'],
            gen_min_span_tree=param['gen_min_span_tree'],
            leaf_size=param['leaf_size'],
            metric=param['metric'],
            min_cluster_size=param['min_cluster_size'],
            min_samples=param['min_samples'],
            p=param['p']
        )

    elif cluster_algorithm == 'agglomerative':
        # Instantiate the agglomerative clusterer.
        default_param = {
            'n_clusters': None,
            'affinity': 'cosine',
            'memory': None,
            'connectivity': None,
            'compute_full_tree': 'auto',
            'linkage': 'single',
            'distance_threshold': 0.55
        }
        param = get_arguments(default_param, parameter_config)
        clusterer = AgglomerativeClustering(
            n_clusters=param['n_clusters'],
            affinity=param['affinity'],
            memory=param['memory'],
            connectivity=param['connectivity'],
            compute_full_tree=param['compute_full_tree'],
            linkage=param['linkage'],
            distance_threshold=param['distance_threshold']
        )

    elif cluster_algorithm == 'optics':
        default_param = {
            'algorithm': 'auto',
            'cluster_method': 'xi',
            'eps': 0.6,
            'predecessor_correction': True,
            'leaf_size': 40,
            'metric': 'euclidean',
            'metric_params': None,
            'min_cluster_size': 30,
            'min_samples': 100,
            'xi': 0.05,
            'n_jobs': None
        }
        param = get_arguments(default_param, parameter_config)

        # Instantiate the OPTICS clusterer.
        clusterer = OPTICS(
            min_samples=param['min_samples'],
            metric=param['metric'],
            metric_params=param['metric_params'],
            cluster_method=param['cluster_method'],
            eps=param['eps'],
            xi=param['xi'],
            predecessor_correction=param['predecessor_correction'],
            min_cluster_size=param['min_cluster_size'],
            algorithm=param['algorithm'],
            leaf_size=param['leaf_size'],
            n_jobs=param['n_jobs']
        )

    elif cluster_algorithm == 'kmeans':

        default_param = {
            'n_clusters': 17,
            'max_iter': 100,
            'init': 'k-means++',
            'n_init': 1
        }
        param = get_arguments(default_param, parameter_config)
        n_clusters = 17
        clusterer = KMeans(
            n_clusters=param['n_clusters'],
           max_iter=param['max_iter'],
           init=param['init'],
           n_init=param['n_init']
        )

    # Fit the clusterer to the data.
    clusterer.fit(clustering_data)
    return clusterer.labels_


def get_weighted_sentence_vectors(sentence_vectors: pd.Series,
                                  sentence_tokens: pd.Series,
                                  word_frequency: Dict,
                                  a=1e-3) -> List:
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
            # Calculate smooth inverse word frequency with tfidf frequency.
            doc_frequency = Counter(tokens)[tok] #/ len(tokens)
            tfidf = word_frequency[tok] * doc_frequency
            a_value = a / (a + tfidf)
            # Adjust new vector according to product of original and frequency.
            new_vector = np.add(new_vector, np.multiply(a_value, vec))

        # Compute weighted average.
        new_vector = np.divide(new_vector, sentence_length)
        sentences.append(new_vector)

    return sentences


def get_word_frequency(comments: pd.Series) -> Dict:
    """
    Get document frequency of each word in the entire corpus.
    - param comments: pd Series
    """
    # Extract all tokens to a single list.
    vocab = list(set([word for comment in comments for word in comment]))

    word_frequency = {}
    # Calculate the frequency across the entire corpus of documents.
    for word in vocab:
        doc_count = 0
        for comment in comments:
            if word in comment:
                doc_count += 1

        word_frequency[word] = math.log(len(comments) / doc_count)

    return word_frequency


def model_topics(data, embeddings, cluster_algorithm, normalization, dim_reduction, outliers, run_grid_search):
    """
    Perform cluster topic modeling using comment mean embeddings.
    - param data: pd dataframe (preprocessed data)
    - param embeddings: str (type of word embeddings)
    - param cluster_algorithm (type of cluster algorithm)
    - param normalization: bool (perform normalization y/n)
    - param dim_reduction: bool (perform dimensionality reduction y/n)
    - param outliers: float (degree of expected dataset contamination)
    - param run_grid_search: bool (perform hyperparamter tuning process)
    """
    # Drop empty values.
    data = data[data['comment_clean'].map(lambda x: len(x) > 0)]

    try:
        # Load embeddings if already calculated.
        print(f'Loading {embeddings} embeddings ...')
        if embeddings == "fasttext":
            embedding_file = "./topic_modeling/topic_modeling_embeddings/_mean_embeddings_fasttext"
        elif embeddings == "bert":
            embedding_file = "./topic_modeling/topic_modeling_embeddings/_mean_embeddings_bert"

        with open(embedding_file, mode="rb") as fp:
            data['embedding'] = _pickle.load(fp)

    except FileNotFoundError as e:
        # Compute embeddings if not calculated.
        print(f'Getting embeddings: {embeddings} ...\n')
        if embeddings == 'fasttext':
            data['embeddings'] = get_fasttext_embeddings(data['comment_clean'])

            # Get mean embeddings.
            print('Computing weighted mean embeddings ...')
            # Compute word frequency for weighted sentence vectors.
            word_frequency = get_word_frequency(data['comment_clean'])
            # Compute sentence embeddings as weighted average of tokens.
            data['embeddings'] = get_weighted_sentence_vectors(data['embeddings'], data['comment_clean'],
                                                               word_frequency)
            # Rename column.
            data.rename(columns={'embeddings': 'embedding'}, inplace=True)
            # Store to accelerate multiple trials.
            print("\nSaving FastText embeddings ...")
            with open("./topic_modeling/topic_modeling_embeddings/_mean_embeddings_fasttext", "wb") as fp:
                _pickle.dump(data['embedding'].tolist(), fp)

        elif embeddings == 'bert':
            # Preprocessing for bert and torch models
            torch_data = bert_preprocessing(data['comment_raw'])
            data['embedding'] = get_bert_embeddings(torch_data)

            print("\nSaving BERT embeddings ...")
            with open("./topic_modeling/topic_modeling_embeddings/_mean_embeddings_bert", mode="wb") as file_handle:
                _pickle.dump(data['embedding'], file_handle)

        else:
            print('Selected embeddings not supported.')
            exit()


    # Apply additional preprocessing.
    clustering_data = np.array(data['embedding'].tolist())

    '''
    this pipeline sets something like a blueprint for
    the whole process. Its architecture should be very
    dynamic and easy to add components:
    each pipeline step gets a name (e.g. normalization)
    each step has three main parts:
    - function: performs step and is designed like the normalize_data
                function so it should take three arguments:
                data, algorithm and parameters in a dictionary
    - parameters: a dictionary which can be empty so the default
                  are set in the function itselfe or it is filled
                  (e.g. for grid search)
    - name: sets algorithm executed in the function
    '''
    pipeline = {}
    if normalization:
        pipeline.update(
            {'normalization':
                 {'function': normalize_data,
                  'parameters': {},
                  'name': 'MinMaxScaler'}
             }
        )
    if dim_reduction:
        pipeline.update(
            {'dim_reduction':
                 {'function':reduce_dimensions,
                  'parameters':{},
                 'name': 'UMAP'}
             }
        )
    pipeline.update(
        {'cluster_algorithm':
             {'function': get_cluster_ids,
              'parameters': {},
              'name': cluster_algorithm}
         }
    )

    # defined by grid search or default in function
    optimal_configurations = dict()

    # in main defined if grid search should be performed
    if run_grid_search:
        # Append normalization step.
        if normalization:
            # Normalize values between 1 and 0.
            pipeline.update(
                {'normalization':
                     {'function': normalize_data,
                      'parameters': {
                          'feature_range': [[0, 1]],
                      },
                      'name': 'MinMaxScaler'}
                 }
            )

        # Append dimensionality reduction and test parameters.
        if dim_reduction:
            # Reduce dimensions for performance while maintaining majority of variance.
            pipeline.update(
                {'dim_reduction':
                     {'function':reduce_dimensions,
                      'parameters':{
                         'metric':['cosine', 'canberra', 'euclidean'],
                         'random_state': [42],
                         'spread': [1, 3, 4, 5, 6],
                         'n_neighbors':[10, 20, 40],
                         'min_dist': [0.0]
                     },
                     'name': 'UMAP'}
                 }
            )

        # Append cluster pipe and test parameters.
        if cluster_algorithm == 'optics':
            # Hyperparameter ranges for optics.
            log_n = int(math.log(clustering_data.shape[0]))
            pipeline.update(
                {'cluster_algorithm':
                     {'function': get_cluster_ids,
                      'parameters': {
                           'leaf_size': [40],
                           'min_samples': [None, 1, log_n],
                           'metric': ['euclidean', 'canberra'],
                           'min_cluster_size': [10, 30, 80, 100],
                           'max_eps': [0.3]
                      },
                      'name': cluster_algorithm}
                 }
            )

        if cluster_algorithm == 'hdbscan':
            # Hyperparameter ranges for hdbscan.
            log_n = int(math.log(clustering_data.shape[0]))
            pipeline.update(
                {'cluster_algorithm':
                     {'function': get_cluster_ids,
                      'parameters': {
                          'alpha': [0.1],
                           'leaf_size': [40],
                           'min_samples': [None, 1, log_n],
                           'metric': ['euclidean', 'canberra'],
                           'min_cluster_size': [10, 30, 80, 100],
                           'cluster_selection_epsilon': [0.3]
                      },
                      'name': cluster_algorithm}
                 }
            )

        if cluster_algorithm == 'kmeans':
            # Hyperparameter ranges for kmeans.
            pipeline.update(
                {'cluster_algorithm':
                    {'function': get_cluster_ids,
                     'parameters': {
                         'n_clusters': [15, 20, 25],
                         'max_iter': [100, 150],
                         'init': ['k-means++'],
                         'n_init': [1, 2]
                       },
                     'name': cluster_algorithm
                     }
                }
            )

        if cluster_algorithm == 'agglomerative':
            # Hyperparameter ranges for agglomerative clustering.
            pipeline.update(
                {'cluster_algorithm':
                    {'function': get_cluster_ids,
                     'parameters': {
                         'affinity': ['cosine', 'euclidean'],
                         'linkage': ['complete', 'average', 'single'],
                         'threshold': [0.1, 0.25, 0.5]
                       },
                     'name': cluster_algorithm
                     }
                }
            )

        # initialize the hyperparameter tuning class
        hyperparameter_tuning = HyperparameterTuning(
            clustering_data,
            pipeline
        )

        # define collection the parameter configurations
        # should be saved in
        MONGO.access_collection('parameter_configurations')

        # performing the search on the parameter grid
        hyperparameter_tuning.perform_grid_search(data_base=MONGO, embedding_type=embeddings)

    if len(optimal_configurations) > 0:
        for component in pipeline:
            pipeline[component]['parameters'].update(optimal_configurations[component])

    # process data with optimal configuration (found by grid search)
    cluster_ids = clustering_data
    for step in pipeline:
        pipeline_component = pipeline[step]['function']
        component_parameters = pipeline[step]['parameters']
        component_name = pipeline[step]['name']
        cluster_ids = pipeline_component(cluster_ids, component_name, component_parameters)
        if step == "dim_reduction":
            data_reduced = cluster_ids
            # Update the dataset with the reduced data for later visualization.
            data['PC1'] = [item[0] for item in data_reduced]
            data['PC2'] = [item[1] for item in data_reduced]

    if outliers and len(data_reduced) != 0:
        # Remove a certain percentage of outliers. Show the number of outliers.
        outlier_scores = LocalOutlierFactor(contamination=outliers).fit_predict(data_reduced)
        cluster_ids = cluster_ids[outlier_scores == 1]
        # Update the dataset to reflect the removed outliers.
        data = data[outlier_scores == 1]

    # Append the cluster ids to the dataframe.
    data['cluster'] = cluster_ids
    n_clusters = data['cluster'].nunique()
    print(f'Found {n_clusters} clusters.')

    return data
