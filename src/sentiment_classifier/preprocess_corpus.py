import xml.etree.ElementTree as ET
import pandas as pd
from src.utilities import preprocessing

def read_corpus(filename, corpus, n):
    """
    Read corpus and return preprocessed documents and sentiment list
    :param filename: Filename of corpus
    :param corpus: Type of corpus
    :param n: Number of documents
    :return: Documents list, sentiments list
    """
    docs = list()
    sentiments = list()
    if corpus == 'germeval':
        root = ET.parse(filename).getroot()
        for doc in root.iter('Document'):
            for text in doc.iter('text'):
                docs.append(text.text)
            for sentiment in doc.iter('sentiment'):
                sentiments.append(sentiment.text)

    elif corpus == 'pnlp':
        data = pd.read_csv(filename, sep=";")

        Question_1 = data[data['Question Text'] == 'Please tell us what is working well.']
        Question_2 = data[data['Question Text'] == 'Please tell us what needs to be improved.']

    elif corpus == 'twitter':
        data = pd.read_csv(filename)
        docs = data['Text'].tolist()
        sentiments = data['Sentiment'].tolist()

    elif corpus == 'imdb':
        data = pd.read_csv(filename)
        print(data.head())
        docs = data['review'].tolist()
        sentiments = data['sentiment'].tolist()

    elif corpus == 'twitter140':
        data = pd.read_csv(filename, encoding="ISO-8859-1",
                           names=["target", "weird number", "date", "query", "user", "text"])
        data = data.sample(frac=1) # ordered dataset -> shuffle beforehand
        docs = data['text'].tolist()
        sentiments = data['target'].tolist()

    elif corpus == 'amazon':
        data = pd.read_csv(filename, encoding='utf-8', sep='\t', header=None)
        data = data.sample(frac=1)
        docs = data[0].tolist()
        sentiments = data[1].tolist()

    elif corpus == 'tomatoes':
        data = pd.read_csv(filename, encoding='ISO-8859-1')
        data = data.sample(frac=1)
        docs = data['Text'].tolist()
        sentiments = data['Rating'].tolist()

    elif corpus == 'german':
        data = pd.read_csv(filename)
        print(data.head())
        docs = data['Text'].tolist()
        sentiments = data['Sentiment'].tolist()

        print("text :", docs[:5])
        print("senti :", sentiments[:5])

    elif corpus == 'airline':
        data = pd.read_csv(filename)
        print(data.head())
        docs = data['text'].tolist()
        sentiments = data['airline_sentiment']

        print("text :", docs[:5])
        print("sentiment :", sentiments[:5])




    return preprocess(docs, sentiments, n)


def preprocess(docs, sentiments, n):
    """
    Filters <br> tags, URLs and twitter handles
    :param docs: Document list
    :param sentiments: Sentiment list
    :param n: Number of documents
    :return: Processed corpus
    """
    processed_tweets = list()
    processed_sentiments = list()


    for i, doc in enumerate(docs):
        if i > n:
            return processed_tweets, processed_sentiments

        if not pd.isna(sentiments[i]):
            processed_tweets.append(doc)
            processed_sentiments.append(str(sentiments[i]))

    processed_tweets = preprocessing(pd.Series(processed_tweets), punctuation=False, correct_apos=True, shortWords=True, specialCharacter=False, singleChar=True).tolist()


    return processed_tweets, processed_sentiments

def read_wordlist():
    """
    Reads sentiment lexicon from file and returns it as dataframe
    """
    wordlist = pd.read_table('Data/SentiWords_1.1.txt', skiprows=30)
    wordlist.columns = ['word', 'sentiment_score']
    wordlist['word'] = wordlist['word'].str.split('#').apply(lambda x: x[0])

    return wordlist

