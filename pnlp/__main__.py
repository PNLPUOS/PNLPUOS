from pnlp.src.topic_modeling.topic_modeling import model_topics
from pnlp.src.utilities import preprocessing, evaluation
from pnlp.src.sentiment_classifier.rnn_sentiment import predict_sentiment
from sacred import Experiment
from sacred.observers import MongoObserver
import pandas as pd
import argparse
import json
import os
import pnlp


pnlp_path = os.path.dirname(pnlp.__file__)

def main(args=None):
    '''
    Main entry point for survey
    analysis pipeline.
    '''
    global config
    with open(f'{pnlp_path}\\config.json', 'r') as f:
        config = json.load(f)
    ex = Experiment()

    url = config['mongodb']['url']
    user = config['mongodb']['username']
    db_name = config['mongodb']['db_name']

    # Optional use sacred observer.
    if len(url) > 0:
        ex.observers.append(MongoObserver(url=url, db_name=db_name))

    # Set configuration from json file.
    preprocessing_param = config['preprocessing']
    topic_model_param = config['topic_model']
    sentiment_param = config['sentiment']
    evaluation_param = config['evaluation']
    data_path = ''


    # Get command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessing', help='Specify as a list the preprocessing steps which \
                            should be performed. These include: \n \
                                \n\t punctuation (remove punctuation)\
                                \n\t correct_spelling (spell correction)\
                                \n\t tokenize (tokenization)\
                                \n\t stopwords (remove stopwords)\
                                \n\t correct_apos (correct apostrophes and contractions)\
                                \n\t shortWords (remove short words)\
                                \n\t specialCharacter (remove special characters)\
                                \n\t numbers (remove numbers)\
                                \n\t singleChar (remove single characters)\
                                \n\t lematization (lemmatize comments)\
                                \n\t stemming (stem comments)\
                                \n\t filter_extremes (remove tokens above and below a certain frequency)\n \
                                \n\nIncluding a parameter means that the processing will be performed. If no \
                                parameters are supplied the default will set all values to True', type=list)
    parser.add_argument('--embeddings', help='Specify as a string the pretrained embedding model to use for document \
                            clustering. Currently the pipeline only supports FastText and Bert models. The \
                            default is "bert"', type=str)
    parser.add_argument('--algorithm', help='Specify the cluster algorith to use for document clustering. \
                            The default is hdbscan.', type=str)
    parser.add_argument('--grid_search', help='If included, the pipeline will run a hyperparameter grid search \
                            over the selected hyperparameters for a set of default ranges.', action='store_true')
    parser.add_argument('--outliers', help='Specify as a float the expected percentage of contamination for outlier \
                            removal. The default is 0.15.', type=float)
    parser.add_argument('--sentiment_model', help='Specify as a string the trained sentiment analysis model \
                            to be used for classification. Options include:\n \
                                \n\t question1 (What is working well.) \
                                \n\t question2 (What can be improved.) \
                                \n\nGiven that each model was trained on questions with an intuitively-biased \
                                sentiment-valence, the chosen model will skew predictions towards either \
                                positive or negative sentiment, respectively.', type=str)
    parser.add_argument('--n_sentences', help='Specify as an integer the number of representative sentences \
                            to return when evaluating the output clusters. The default is 3.', type=int)
    parser.add_argument('--keywords', help='Specify as a string the method for keyword extraction. Options include \
                            "tfidf" and "frequency." The default is "frequency"', type=str)
    parser.add_argument('--labels', help='Specify as a string the method for generating topic labels. Options \
                            include "mean_projection" and "top_5_words." If "top_5_words," the pipeline will \
                            select the top five words based on the selected keyword-extraction method. The \
                            "mean_projection" method is only available when using FastText vectors.', type=str)
    parser.add_argument('--method_sentences', help='Specify as a string the method for representative \
                            sentence extraction. Options include "statistical" and "embedding". The \
                            default is "embedding."', type=str)
    parser.add_argument('--path', help='Specify as a string the path to your data.', type=str)
    args = parser.parse_args()

    # Store the path.
    data_path = vars(args)["path"]
    # Override default arguments if provided.
    if vars(args)["preprocessing"]:
        for item in preprocessing:
            preprocessing_param[item] = True
    if vars(args)["embeddings"]:
        topic_model_param['embeddings'] = vars(args)["embeddings"]
    if vars(args)["algorithm"]:
        topic_model_param['cluster_algorithm'] = vars(args)["algorithm"]
    if vars(args)["grid_search"]:
        topic_model_param['run_grid_search'] = True
    if vars(args)["outliers"]:
        topic_model_param['outliers'] = vars(args)["outliers"]
    if vars(args)["sentiment_model"]:
        sentiment_param['load'] = vars(args)["sentiment_model"]
    if vars(args)["n_sentences"]:
        evaluation_param['n_sentences'] = vars(args)["n_sentences"]
    if vars(args)["keywords"]:
        evaluation_param['keywords'] = vars(args)["keywords"]
    if vars(args)["labels"]:
        evaluation_param['labels'] = vars(args)["labels"]
    if vars(args)["method_sentences"]:
        evaluation_param['method_sentences'] = vars(args)["method_sentences"]

    # TODO: Print summary of configuration

    @ex.config
    def configuration():
        '''
        Configure experiment parameters to be logged
        by sacred.
        '''
        pass

    @ex.main
    def run(experimenter=user, data_path=data_path,
                preprocessing_param=preprocessing_param, topic_model_param=topic_model_param,
                    evaluation_param=evaluation_param, sentiment_param=sentiment_param, ex=ex):
        # Load raw data.
        df_raw = pd.read_csv(data_path, delimiter=';')
        # Extract only the comments.
        series = df_raw['Comments']
        # Preprocessing.
        data = preprocessing(series, **preprocessing_param).to_frame().rename(columns={"Comments": "comment_clean"})
        # Append raw comments needed for specific methods.
        data['comment_raw'] = series

        # Ignore other columns if not provided.
        if ('Report Grouping' and 'Question Text') in df_raw.columns:
            # Add other original columns.
            data['Report Grouping'] = df_raw['Report Grouping']
            data['Question Text'] = df_raw['Question Text']

        # Topic modeling.
        data = model_topics(data, **topic_model_param)
        # Append sentiment information.
        data['sentiment'] = predict_sentiment(data['comment_raw'], **sentiment_param)
        # Evaluate the results.
        data_path, clusters_path, graph_path = evaluation(data, **evaluation_param)
        # TODO: Log information to sacred.
        # ex.log_scalar('n_clusters', data['cluster'].nunique())
        # ex.add_artifact(clusters_path)

        print('Analysis complete.')
        print('Please check the outputs directory or visit omniboard to view the experiment results.')

    run()

if __name__ == "__main__":
    main()
