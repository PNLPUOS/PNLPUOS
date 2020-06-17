# pnlp
from topic_modeling.topic_modeling import model_topics
from utilities import preprocessing, evaluation
from sentiment_classifier import *
from credentials import username, password

# sacred, misc
from sacred import Experiment
from sacred.observers import MongoObserver
import pandas as pd

# Instantiate the sacred experiment.
ex = Experiment()
db_name = 'pnlp'
url = f'mongodb+srv://{username}:{password}@cluster0-8ejtu.azure.mongodb.net/{db_name}?retryWrites=true&w=majority&ssl=true&ssl_cert_reqs=CERT_NONE'

# Add the MongoDB observer.
# ex.observers.append(MongoObserver(url=url, db_name=f'{db_name}'))

# Configure the experiment parameters. These will be logged by sacred.
@ex.config
def config():
    experimenter = username
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
            'run_grid_search': False

    }
    sentiment = False
    evaluation_param = {

            'keywords': 'frequency',
            'labels': 'top_5_words',
            'method_sentences': 'embedding',
            'n_sentences': 3

    }


# Sacred will run the main loop from the terminal.
@ex.automain
def run(experimenter, data_path, data_language, preprocessing_param, topic_model_param, evaluation_param, sentiment):
    # Load raw data.
    series = pd.read_csv(data_path, delimiter=';')['Comments']
    # Extract first or second question only.
    # series = series[series['Question Text'] == 'Please tell us what is working well.']['Comments']
    # series = series[series['Question Text'] == 'Please tell us what needs to be improved.']['Comments']
    # Preprocessing.
    data = preprocessing(series, **preprocessing_param).to_frame().rename(columns={"Comments": "comment_clean"})
    # Append raw comments needed for specific methods.
    data['comment_raw'] = series
    # Topic modeling.
    data = model_topics(data, **topic_model_param)
    # Evaluate the results.
    data_path, clusters_path, graph_path = evaluation(data, **evaluation_param)
    # Log information to sacred.
    ex.log_scalar('n_clusters', data['cluster'].nunique())
    ex.add_artifact(clusters_path)

    print(f'Please visit omniboard to view the experiment results. You may need to gain access to the MongoDB.\n\tURL: {url}')
