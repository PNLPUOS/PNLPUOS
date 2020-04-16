import numpy as np
from gensim.models import KeyedVectors
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, GRU
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras import callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import pandas as pd
from nltk import tokenize, TweetTokenizer

from preprocess_corpus import read_corpus


EMBED_DIM = 100       # dimension of embedding matrix
MAX_FEATURES = 20000   # maximum number of words in the vocabulary, will take the 5000 most common words in training data
MAX_LEN = 30          # max length of each document, shorter documents are padded, longer ones are cut short
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


def encode_classes(Y_true):
    """
    Transforms categorical labels into integer matrix for Neural Network
    :param Y_true: Predicted labels (strings)
    :return: Integer matrix
    """
    encoder = LabelEncoder()
    Y_int = encoder.fit_transform(Y_true)
    #print('encoding: ')
    print(encoder.classes_)
    #print('to_categorical: ',to_categorical(Y_int[:10]))

    return to_categorical(Y_int)


def build_LSTM_model(vocab_size, embedding_matrix=np.empty((0))):
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

    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(CLASS_NUM, activation='softmax'))
    #model.add(Dense(1, activation='softmax'))
    adam = Adam(lr=1e-3, epsilon=1e-8, beta_1=.9, beta_2=.999)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())

    return model


def build_GRU_model(vocab_size, embedding_matrix=np.empty((0))):
    """
    Builds GRU model for sentiment analysis
    :param vocab_size: Size of vocabulary
    :param embedding_matrix: Matrix containing pre-trained word embeddings, optional
    :return: GRU model
    """
    model = Sequential()

    if embedding_matrix.size == 0:
        model.add(Embedding(vocab_size + 1, EMBED_DIM, input_length=MAX_LEN))
    else:
        model.add(
            Embedding(vocab_size + 1, EMBED_DIM, weights=[embedding_matrix], input_length=X_train.shape[1], trainable=False))

    model.add(SpatialDropout1D(0.4))
    model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(CLASS_NUM, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    return model


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
    w2v_model = KeyedVectors.load_word2vec_format(filename, binary=False, encoding='utf-8')
    return get_embedding_matrix(w2v_model.wv, vocab)



if __name__ == '__main__':


    #Read and preprocess data

    data = pd.read_csv('sentiment_dataset.csv')

    Question_1 = data[data['question'] == 'Please tell us what is working well.']
    Question_2 = data[data['question'] == 'Please tell us what needs to be improved.']


    #For testing purposes I'm only looking at the first question

    documents, sentiments = read_corpus('data/sentiment_corpus_twitter.txt', 'twitter', 1000000)

    #documents, sentiments = Question_1['comments'], Question_1['sentiment']

    CLASS_NUM = len(set(sentiments))
    print('numclasses: ',CLASS_NUM)
    print('length of documents: ',len(documents))
    print('length of sentiments: ',len(sentiments))
    print('first 10 doc: ',documents[:10])
    print('first 10 sent: ',sentiments[:10])
    #print('docs',documents)

    #X, tokenizer = preprocess(documents)
    #Y = encode_classes(sentiments)

    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


    X_train,tokenizer = preprocess(documents)
    Y_train = encode_classes(sentiments)
    vocabulary = tokenizer.word_index
    #X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.25, random_state=42)

    #X_test, _ = preprocess(Question_1['comments'], tokenizer)
    #Y_test = encode_classes(Question_1['sentiment'])

    #X_2, _ = preprocess(Question_2['comments'], tokenizer)
    #bla, _ = preprocess(Question_1['comments'], tokenizer)
    X_2, _ = preprocess(data['comments'], tokenizer)
    #Y_2 = encode_classes(Question_2['sentiment'])
    Y_2 = encode_classes(data['sentiment'])
    X_train_2, X_test, Y_train_2, Y_test = train_test_split(X_2, Y_2, test_size=0.1, random_state=42)

    print('Y_train encoded classes: ', sentiments[:10])
    print('Y_test encoded classes: ', Question_1['sentiment'].head(10))



    #print('shape xtest: ',X_test.shape)


    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=32)



    #print('length of x train: ',len(X_train),'len of y train: ',len(Y_train))

    embeddings = False
    if embeddings:  # build model with pre-trained embeddings
        embedding_matrix = get_embeddings(vocabulary, 'Embeddings/embed_tweets_en_590M_52D')
        model = build_LSTM_model(len(vocabulary), embedding_matrix)
    else:       # build model and train own embeddings on training data
        model = build_LSTM_model(len(vocabulary))

    # stop training process early if the validation loss stops decreasing
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
    callback_list = [early_stopping]

    model.fit(X_train, Y_train, epochs=10, batch_size=64, verbose=2, validation_data=(X_test, Y_test),
              callbacks=callback_list)


    print("second run with new data: ")
    print("----------")

    model.fit(X_train_2,Y_train_2, epochs=10, batch_size=64, verbose=2, validation_data=(X_test, Y_test))

    #for x in X_test:
    #    pred = model.predict(x)
    #    print('Comment: ', x, 'prediction: ', pred)

    # Evaluate
    score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=32)
    print(score, acc)
    negative = 0
    neutral = 0
    positive = 0

    neg_pred = 0
    neu_pred = 0
    pos_pred = 0

    for i,x in enumerate(model.predict_classes(X_test)):
        #print(x)
        actual = np.argmax(Y_test[i])
        #print(actual)
        if x== 0:
            neg_pred += 1
        elif x==1:
            neu_pred += 1
        elif x==2:
            pos_pred += 1
        if actual== 0:
            negative += 1
        elif actual==1:
            neutral += 1
        elif actual==2:
            positive += 1

    print('number of negative tweets: ',negative,'num of negative predictions', neg_pred)
    print('number of neutral tweets: ',neutral,'num of neutral predictions', neu_pred)
    print('number of positive tweets: ',positive,'num of positive predictions', pos_pred)
