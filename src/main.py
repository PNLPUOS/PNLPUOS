# pnlp
from topic_modeling import model_topics
from utilities import preprocessing, evaluation
from sentiment_classifier import *
from credentials import username, password

# sacred, misc
from sacred import Experiment
from sacred.observers import MongoObserver
import pandas as pd


ex = Experiment()
db_name = 'pnlp'

# ex.observers.append(MongoObserver(url=f'mongodb+srv://{username}:{password}@cluster0-8ejtu.azure.mongodb.net/{db_name}?retryWrites=true&w=majority',
#                                   db_name=f'{db_name}'))

@ex.config
def config():
    experimenter = 'Christian'
    data_path = '../data/pnlp_data_en.csv'
    data_language = 'english'
    preprocessing_param = {

            'punctuation': True,
            'tokenize': True,
            'stopwords': True,
            'correct_apos': True,
            'shortWords': True,
            'specialCharacter': True,
            'numbers': True,
            'singleChar': True,
            'lematization': True,
            'stemming': False,

    }
    topic_model_param = {

            'embeddings': 'fasttext',
            'cluster_algorithm': 'hdbscan',
            'normalization': True,
            'dim_reduction': True,
            'outliers': 0.15,

    }
    sentiment = False
    evaluation_param = {

            'keywords': 'tfidf',
            'labels': 'top_5_words',

    }


@ex.automain
def run(experimenter, data_path, data_language, preprocessing_param, topic_model_param, evaluation_param, sentiment):
    series = pd.read_csv(data_path, delimiter=';')['Comments']
    data = preprocessing(series, **preprocessing_param).to_frame().rename(columns={"Comments": "comment"})
    data = model_topics(data, **topic_model_param)
    data_path, clusters_path, graph_path = evaluation(data, **evaluation_param)

    # Log information to sacred.
    ex.log_scalar('n_clusters', data['cluster'].nunique())
    # TODO: ex.log_scalar(custom_eval_metric)
    ex.add_artifact(clusters_path)

    print('Please visit omniboard to view the experiment results. You may need to gain access to the MongoDB.')
