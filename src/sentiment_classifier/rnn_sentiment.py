import numpy as np
from gensim.models import KeyedVectors
from keras.models import load_model
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, GRU, Concatenate, Input
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras import callbacks, Model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
from nltk import tokenize, TweetTokenizer

from src.sentiment_classifier.preprocess_corpus import read_corpus

EMBED_DIM = 52  # dimension of embedding matrix
MAX_FEATURES = 20000  # maximum number of words in the vocabulary, will take the 5000 most common words in training data
MAX_LEN = 30  # max length of each document, shorter documents are padded, longer ones are cut short
CLASS_NUM = 3


def preprocess(docs, tok=None):
    """
    Transforms documents for Neural Network
    :param docs: Array of strings (documents)
    :param tok: Tokenizer with already trained vocabulary, optional
    :return: transformed matrix, tokenizer
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
    :param Y_true: Predicted labels (strings)
    :return: Integer matrix
    """
    encoder = LabelEncoder()
    if numerical:
        encoder.fit(['-1', '0', '1'])
    else:
        encoder.fit(['negative', 'neutral', 'positive'])
    Y_int = encoder.transform(Y_true)


    return to_categorical(Y_int, 3)


def build_LSTM_model(lstm_features_test, x_train):
    feature_input = Input(shape=(lstm_features_test.shape[1],), name='aux_input')
    feature_dense2 = Dense(64, activation='relu')(feature_input)
    feature_dense4 = Dense(128, activation='relu')(feature_dense2)

    embed_input = Input(shape=(x_train.shape[1],), name='main_input')
    embedding = Embedding(MAX_FEATURES, EMBED_DIM)(embed_input)
    hidden1 = LSTM(300, dropout=0.2, recurrent_dropout=0.2)(embedding)

    x = Concatenate(axis=1)([hidden1, feature_dense4])
    main_output = Dense(3, activation='softmax', name='main_output')(x)
    model = Model(inputs=[embed_input, feature_input], output=main_output)

    adam = Adam(lr=1e-3, epsilon=1e-8, beta_1=.9, beta_2=.999)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

def build_LSTM_model2(vocab_size, embedding_matrix=np.empty((0))):
    """
    Builds LSTM model for sentiment analysis
    :param vocab_size: Size of vocabulary
    :param embedding_matrix: Matrix containing pre-trained word embeddings, optional
    :return: LSTM model
    """
    lstm_out = 196
    model = Sequential()

    if embedding_matrix.size == 0:
        model.add(Embedding(MAX_FEATURES, EMBED_DIM))
    else:
        model.add(
            Embedding(vocab_size + 1, EMBED_DIM, weights=[embedding_matrix], input_length=X_train.shape[1], trainable=False))

    model.add(SpatialDropout1D(0.1))
    model.add(LSTM(lstm_out, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(CLASS_NUM, activation='softmax'))
    #model.add(Dense(1, activation='softmax'))
    adam = Adam(lr=1e-3, epsilon=1e-8, beta_1=.9, beta_2=.999)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


def tfidf_feature(docs, vocab=None):
    print('tf-idf feature...')
    if vocab is None:
        vectorizer = TfidfVectorizer(analyzer='word', tokenizer=lambda x: x.split(' '))
    else:
        vectorizer = TfidfVectorizer(analyzer='word', vocabulary=vocab,
                                     tokenizer=lambda x: x.split(' '))
    tfidf = vectorizer.fit_transform(docs).toarray()

    return tfidf, vectorizer.vocabulary_


def get_avg_embedding(embedding, vocab):
    weights = list()
    for word, i in vocab.items():
        if embedding.__contains__(word):
            weights.append(embedding.get_vector(word))

    return np.average(weights, axis=0)


def get_embedding_matrix(embedding, vocab):
    avg_embedding = get_avg_embedding(embedding, vocab)
    vocab_size = len(vocab) + 1
    weight_matrix = np.zeros((vocab_size, EMBED_DIM))

    for word, i in vocab.items():
        if embedding.__contains__(word):
            weight_matrix[i] = embedding.get_vector(word)
        else:
            weight_matrix[i] = avg_embedding

    return weight_matrix


def get_embeddings(vocab, filename):
    """
    Creates an embedding matrix from pre-trained word embeddings in word2vec format
    :param vocab: Vocabulary of the tokenizer
    :param filename: Filename/path to the pre-trained embeddings
    :return: The embedding matrix
    """
    print("Loading FastText model...")  # https://www.spinningbytes.com/resources/wordembeddings/
    w2v_model = KeyedVectors.load_word2vec_format(filename, binary=True, encoding='utf-8')
    return get_embedding_matrix(w2v_model.wv, vocab)


def predict_sentiment(docs, out_path, tokenizer=None, model=None, voc=None, load=False):
    if load and isinstance(load, str):
        print("Loading model {}".format(load))
        try:
            model = load_model("data/{}_trained_model.h5".format(load))
            tokenizer = pickle.load(open('data/{}_trained_tokenizer.pickle'.format(load), 'rb'))
        except FileNotFoundError:
            print('Requested model could not be found and loaded.')
    if not tokenizer:
        try:
            print("Loading pre-existing tokenizer")
            with open('data/sentiment_tokenizer_trained.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
        except FileNotFoundError:
            print("No tokenizer could be found. ")
    if not model:
        try:
            print("Loading sentiment analysis model...")
            model = load_model('data/sentiment_model_trained.h5')
        except FileNotFoundError:
            print("No model could be found.")
    if not voc:
        try:
            #print("Loading pre-existing tf-idf vectorizer")
            with open('data/sentiment_tf_idf_vocab.pickle', 'rb') as handle:
                voc = pickle.load(handle)
        except FileNotFoundError:
            pass
            #print("No tf-idf vectorizer could be found")

    #docs_tfidf, _ = tfidf_feature(docs.values.tolist(), voc)
    processed_docs, _ = preprocess(docs.values.tolist(), tokenizer)
    #predictions = model.predict([processed_docs, docs_tfidf])
    predictions = model.predict(processed_docs)
    sentiment = []
    for pred in predictions:
        sentiment.append(pred.tolist().index(max(pred)) - 1)
    docs = docs.to_frame()
    docs['labels'] = sentiment
    docs.to_csv(out_path, index=None)



def train_sentiment_classifier(data, embeddings=False, save=False):
    twitter, sentiment = read_corpus('Data/Tweets.csv', 'airline', 15000)
    twitter_texts, tokenizer = preprocess(twitter)
    twitter_sentiment = encode_classes(sentiment, numerical=False)
    vocabulary = tokenizer.word_index
    X = data['comments']
    Y = data['sentiment']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


    twitter_tfidf, tfidf_vocab = tfidf_feature([' '.join(comment) for comment in twitter])
    X_train_tfidf, _ = tfidf_feature([' '.join(comment) for comment in X_train], tfidf_vocab)
    X_test_tfidf, _ = tfidf_feature([' '.join(comment) for comment in X_test], tfidf_vocab)





    X_train, _ = preprocess(X_train, tokenizer)
    X_test, _ = preprocess(X_test, tokenizer)
    Y_train = encode_classes(Y_train)
    Y_test = encode_classes(Y_test)

    if embeddings:  # build model with pre-trained embeddings
        embedding_matrix = get_embeddings(vocabulary, 'D:/PNLP/Embeddings/embed_tweets_en_590M_52D')
        model = build_LSTM_model(len(vocabulary), embedding_matrix)
    else:  # build model and train own embeddings on training data
        #model = build_LSTM_model2(X_test_tfidf, X_train)
        model = build_LSTM_model2(X_train)

    # stop training process early if the validation loss stops decreasing
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    callback_list = [early_stopping]

    #model.fit([twitter_texts, twitter_tfidf], twitter_sentiment, verbose=2, batch_size=32, epochs=10,
    #          validation_data=([X_test, X_test_tfidf], Y_test))
    model.fit(twitter_texts, twitter_sentiment, verbose=2, batch_size=32, epochs=10, validation_data=(X_test, Y_test))

    print("second run with new data: ")
    print("----------")

    model.fit(X_train, Y_train, verbose=2, batch_size=32, epochs=10,
              validation_data=(X_test, Y_test), callbacks=callback_list)

    # for x in X_test:
    #    pred = model.predict(x)
    #    print('Comment: ', x, 'prediction: ', pred)

    # # Evaluate
    #score, acc = model.evaluate([X_test, X_test_tfidf], Y_test, verbose=2, batch_size=32)
    score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=32)
    print('score: ', score)
    print('acc: ', acc)

    negative = 0
    neutral = 0
    positive = 0

    neg_pred = 0
    neu_pred = 0
    pos_pred = 0

    for i, x in enumerate(model.predict(X_test)):
    #for i, x in enumerate(model.predict([X_test, X_test_tfidf])):
        x = np.argmax(x)
        # print(x)
        actual = np.argmax(Y_test[i])
        # print(actual)
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
        model.save('Data/{}_trained_model.h5'.format(save))
        pickle.dump(tokenizer, open('Data/{}_trained_tokenizer.pickle'.format(save), "wb"))
        #pickle.dump(tfidf_vocab, open('data/sentiment_tf_idf_vocab.pickle', 'wb'))

    return model, tokenizer, tfidf_vocab


if __name__ == '__main__':
    # Read and preprocess the training data
    data = pd.read_csv('Data/sentiment_dataset.csv', names=['id', 'grouping', 'question', 'sentiment', 'comments'])

    question_1 = data[data['question'] == 'Please tell us what is working well.']
    question_2 = data[data['question'] == 'Please tell us what needs to be improved.']
    # Train the model on separate questions
    #mod, tok = train_sentiment_classifier(question_2, save=False')

    # Load unlabeled data to label with sentiment
    test_data = pd.read_csv('D:/PNLP/Data/pnlp_data_en.csv', sep=';', names=['grouping', 'question', 'comments'])
    test_question_1 = test_data[test_data['question'] == 'Please tell us what is working well.']
    test_question_2 = test_data[test_data['question'] == 'Please tell us what needs to be improved.']
    # Sample a random number of comments from one question to label them
    random_subset = test_question_2['comments'].sample(n=100)

    # Load pre-existing model trained on question 1 and label the dataset
    predict_sentiment(random_subset, 'data/predicted_sentiments.csv', load='question2')
    # TODO return sentiment labels as a Series from predict_sentiment instead of saving file
    # TODO preprocess text within the method (pred_sent..)
    # TODO train classifier with question as feature