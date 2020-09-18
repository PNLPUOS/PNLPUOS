import numpy as np
from keras.models import load_model
from keras.layers import Dense, Embedding, LSTM, Concatenate, Input
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras import callbacks, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
import os

from pnlp.src.sentiment_classifier.preprocess_corpus import read_corpus
from pnlp.src.sentiment_classifier.preprocess_corpus import read_wordlist
from pnlp.src.utilities import preprocessing

pnlp_path = os.path.dirname(pnlp.__file__) # path to the local pnlp directory
EMBED_DIM = 52  # dimension of embedding matrix
MAX_FEATURES = 20000  # maximum number of words in the vocabulary, will take the 5000 most common words in training data
MAX_LEN = 30  # max length of each document, shorter documents are padded, longer ones are cut short
CLASS_NUM = 3
WORDLIST = read_wordlist()


def preprocess(docs, tok=None):
    """
    Transforms documents for Neural Network
    Params:
        docs = Array of strings (documents)
        tok = Tokenizer with already trained vocabulary, optional
    Returns:
         transformed matrix, tokenizer
    """
    if tok is None:
        tokenizer = Tokenizer(num_words=MAX_FEATURES, split=' ', lower=True, oov_token='<oov>')
        tokenizer.fit_on_texts(docs)  # Updates internal vocabulary based on a list of texts

    else:
        tokenizer = tok

    X = tokenizer.texts_to_sequences(docs)  # Transforms each text in texts to a sequence of integers

    X = pad_sequences(X, maxlen=MAX_LEN, padding='post')

    return X, tokenizer


def encode_classes(Y_true, numerical=True):
    """
    Transforms categorical labels into integer matrix for Neural Network
    Params:
        Y_true = Predicted labels (strings)
        numerical = False, if labels are not numerical
    Returns:
        Integer matrix
    """
    encoder = LabelEncoder()
    if numerical:
        encoder.fit(['-1', '0', '1'])
    else:
        encoder.fit(['negative', 'neutral', 'positive'])
    Y_int = encoder.transform(Y_true)

    return to_categorical(Y_int, 3)


def build_LSTM_model(lstm_features_train, x_train):
    """
    Build model architecture with 2 inputs
    Params:
        lstm_features_train = Sentiment lexicon feature matrix
        x_train = matrix of input texts in numerical representation
    Returns:
        model with initialized layers
    """
    # numerical feature input (aux_input)
    feature_input = Input(shape=(lstm_features_train.shape[1],), name='aux_input')
    feature_dense2 = Dense(64, activation='relu')(feature_input)
    feature_dense4 = Dense(128, activation='relu')(feature_dense2)

    # text input for lstm (main_input)
    embed_input = Input(shape=(x_train.shape[1],), name='main_input')
    embedding = Embedding(MAX_FEATURES, EMBED_DIM)(embed_input)
    hidden1 = LSTM(300, dropout=0.2, recurrent_dropout=0.2)(embedding)

    # concatenate inputs and apply optimizer
    x = Concatenate(axis=1)([hidden1, feature_dense4])
    main_output = Dense(3, activation='softmax', name='main_output')(x)
    model = Model(inputs=[embed_input, feature_input], output=main_output)

    adam = Adam(lr=1e-3, epsilon=1e-8, beta_1=.9, beta_2=.999)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


def sentiment_lexicon_feature(docs):
    """
    Generate sentiment lexicon features from the dataset
    The features are based on the SentiWords sentiment lexicon where each word present has an associated sentiment
    score ranging from -1 to 1
    The sentiment feature matrix for each comment consists of the cumulative positive and negative score of all words in the text,
    as well as number of positive and negative words

    """
    print('Sentiment word list feature...')
    wordlist_feature = np.zeros((len(docs), 4))

    for i, comment in enumerate(docs):
        n_neg = 0
        n_pos = 0
        neg = 0
        pos = 0
        for token in comment:
            if token in WORDLIST['word']:
                sentiment_score = WORDLIST.loc[WORDLIST['word'] == token, 'sentiment_score'].iloc[0]
                if sentiment_score < 0:
                    neg += sentiment_score
                    n_neg += 1
                if sentiment_score > 0:
                    pos += sentiment_score
                    n_pos += 1

        wordlist_feature[i][0] = neg
        wordlist_feature[i][1] = pos
        wordlist_feature[i][2] = n_neg / len(comment)
        wordlist_feature[i][3] = n_pos / len(comment)

    return wordlist_feature


def get_avg_embedding(embedding, vocab):
    weights = list()
    for word, i in vocab.items():
        if embedding.__contains__(word):
            weights.append(embedding.get_vector(word))

    return np.average(weights, axis=0)


def predict_sentiment(docs, out_path=None, tokenizer=None, model=None, load=False):
    """
    Predict the sentiments of a list of documents. Can load a pre-existing model if the path is given,
    otherwise a model and a tokenizer must be provided as argument.

    Params:
        out_path = output path for the .csv file with the comments and predicted labels
        tokenizer = name of the tokenizer to be loaded
        model = name of the trained model to be loaded
        load = if True, load the tokenizer and the model
    Return:
        list of sentiment labels. If output path is given, the results will be saved in a dataframe
    """
    if load and isinstance(load, str):
        print("Loading model {}".format(load))
        try:
            model = load_model(f"{pnlp_path}/src/sentiment_classifier/data/{load}_trained_model.h5")
            tokenizer = pickle.load(open(f'{pnlp_path}/src/sentiment_classifier/data/{load}_trained_tokenizer.pickle', 'rb'))
            print("Model loaded")
        except FileNotFoundError:
            print('Requested model could not be found and loaded.')
    if not tokenizer:
        try:
            print("Loading pre-existing tokenizer")
            with open(f'{pnlp_path}/src/sentiment_classifier/data/sentiment_tokenizer_trained.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
        except FileNotFoundError:
            print("No tokenizer could be found. ")
    if not model:
        try:
            print("Loading sentiment analysis model...")
            model = load_model(f'{pnlp_path}/src/sentiment_classifier/data/sentiment_model_trained.h5')
        except FileNotFoundError:
            print("No model could be found.")

    # pre-process the comments (basic text clean-up)
    preprocessed_docs = preprocessing(pd.Series(docs), punctuation=False, correct_apos=True, shortWords=True,
                                      specialCharacter=False, singleChar=True).tolist()
    # generate sentiment lexicon features
    docs_sentiment_lexicon = sentiment_lexicon_feature(preprocessed_docs)
    # transforms the comments into numeric representations to be used for the neural network
    processed_docs, _ = preprocess(preprocessed_docs, tokenizer)
    # generate the sentiment predictions and assign them to the corresponding text
    predictions = model.predict([processed_docs, docs_sentiment_lexicon])
    sentiment = []
    for pred in predictions:
        sentiment.append(pred.tolist().index(max(pred)) - 1)
    docs = docs.to_frame()
    docs['labels'] = sentiment
    docs.to_csv(out_path, index=None)
    return docs['labels']


def train_sentiment_classifier(data, save=None):
    """
    Train a sentiment classifier from a labeled dataset. Uses an external dataset ("Tweets.csv) to
    generate word embeddings and tokenizer to augment small dataset input
    """
    twitter, sentiment = read_corpus(f'{pnlp_path}/src/sentiment_classifier/data/Tweets.csv', 'airline', 15000)
    twitter_texts, tokenizer = preprocess(twitter)
    twitter_features_lexicon = sentiment_lexicon_feature(twitter)
    twitter_sentiment = encode_classes(sentiment, numerical=False)
    X = data['comments']
    Y = data['sentiment']
    # split dataset into training and testing data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    # generate sentiment lexicon features
    X_train_lexicon = sentiment_lexicon_feature(X_train)
    X_test_lexicon = sentiment_lexicon_feature(X_test)

    # convert input texts and labels to numerical representations
    X_train, _ = preprocess(X_train, tokenizer)
    X_test, _ = preprocess(X_test, tokenizer)
    Y_train = encode_classes(Y_train)
    Y_test = encode_classes(Y_test)

    # create model architecture, initialize layers, etc.
    model = build_LSTM_model(X_train_lexicon, X_train)

    # stop training process early if the validation loss stops decreasing
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    callback_list = [early_stopping]

    # train the model on the twitter airline corpus
    model.fit([twitter_texts, twitter_features_lexicon], twitter_sentiment, verbose=2, batch_size=32, epochs=10,
              validation_data=([X_test, X_test_lexicon], Y_test))

    print("second run with new data: ")
    print("----------")

    # fit the trained model on the data given by the user
    model.fit([X_train, X_train_lexicon], Y_train, verbose=2, batch_size=32, epochs=10,
              validation_data=([X_test, X_test_lexicon], Y_test), callbacks=callback_list)

    # # Evaluate and return statistics about the predicted labels
    score, acc = model.evaluate([X_test, X_test_lexicon], Y_test, verbose=2, batch_size=32)
    print('score: ', score)
    print('acc: ', acc)

    negative = 0
    neutral = 0
    positive = 0

    neg_pred = 0
    neu_pred = 0
    pos_pred = 0

    for i, x in enumerate(model.predict([X_test, X_test_lexicon])):
        x = np.argmax(x)
        actual = np.argmax(Y_test[i])
        if x == 0:
            neg_pred += 1
        elif x == 1:
            neu_pred += 1
        elif x == 2:
            pos_pred += 1
        if actual == 0:
            negative += 1
        elif actual == 1:
            neutral += 1
        elif actual == 2:
            positive += 1

    print('number of negative tweets: ', negative, 'num of negative predictions', neg_pred)
    print('number of neutral tweets: ', neutral, 'num of neutral predictions', neu_pred)
    print('number of positive tweets: ', positive, 'num of positive predictions', pos_pred)

    if save and isinstance(save, str):
        model.save(f'{pnlp_path}/src/sentiment_classifier/data/{save}_trained_model.h5')
        pickle.dump(tokenizer, open(f'{pnlp_path}/src/sentiment_classifier/data/{save}_trained_tokenizer.pickle', "wb"))

    return model, tokenizer
