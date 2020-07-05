# nltk
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
#from porter2stemmer import Porter2Stemmer
import spell_correction.spellcorrector as spellcorrector

# html
from bs4 import BeautifulSoup
from html.parser import HTMLParser

# data
import pandas as pd
import numpy as np
import scipy as sp

# visualization
import seaborn as sns

# misc
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import fasttext
import annoy
import os
import re
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
import logging
logging.getLogger("gensim.models").setLevel(logging.WARNING)


stopwordList = stopwords.words('english')

# Apostrophe Dictionary and if have more words in mind, please add it in the bottom
apostrophe = {
    "aren't" : "are not",
    "can't" : "cannot",
    "couldn't" : "could not",
    "didn't" : "did not",
    "doesn't" : "does not",
    "don't" : "do not",
    "hadn't" : "had not",
    "hasn't" : "has not",
    "haven't" : "have not",
    "he'd" : "he would",
    "he'll" : "he will",
    "he's" : "he is",
    "i'd" : "I would",
    "i'd" : "I had",
    "i'll" : "I will",
    "i'm" : "I am",
    "isn't" : "is not",
    "it's" : "it is",
    "it'll":"it will",
    "i've" : "I have",
    "let's" : "let us",
    "mightn't" : "might not",
    "mustn't" : "must not",
    "shan't" : "shall not",
    "she'd" : "she would",
    "she'll" : "she will",
    "she's" : "she is",
    "shouldn't" : "should not",
    "that's" : "that is",
    "there's" : "there is",
    "they'd" : "they would",
    "they'll" : "they will",
    "they're" : "they are",
    "they've" : "they have",
    "we'd" : "we would",
    "we're" : "we are",
    "weren't" : "were not",
    "we've" : "we have",
    "what'll" : "what will",
    "what're" : "what are",
    "what's" : "what is",
    "what've" : "what have",
    "where's" : "where is",
    "who'd" : "who would",
    "who'll" : "who will",
    "who're" : "who are",
    "who's" : "who is",
    "who've" : "who have",
    "won't" : "will not",
    "wouldn't" : "would not",
    "you'd" : "you would",
    "you'll" : "you will",
    "you're" : "you are",
    "you've" : "you have",
    "'re": " are",
    "wasn't": "was not",
    "we'll":" will",
    "didn't": "did not",
    "'s": "is",
    "'re": "are"
}

#Short words dictionary and if have more words in mind, please add it in the bottom
short_words = {
"121": "one to one",
"a/s/l": "age, sex, location",
"adn": "any day now",
"afaik": "as far as I know",
"afk": "away from keyboard",
"aight": "alright",
"alol": "actually laughing out loud",
"b4": "before",
"b4n": "bye for now",
"bak": "back at the keyboard",
"bf": "boyfriend",
"bff": "best friends forever",
"bfn": "bye for now",
"bg": "big grin",
"bta": "but then again",
"btw": "by the way",
"cid": "crying in disgrace",
"cnp": "continued in my next post",
"cp": "chat post",
"cu": "see you",
"cul": "see you later",
"cul8r": "see you later",
"cya": "bye",
"cyo": "see you online",
"dbau": "doing business as usual",
"fud": "fear, uncertainty, and doubt",
"fwiw": "for what it's worth",
"fyi": "for your information",
"g": "grin",
"g2g": "got to go",
"ga": "go ahead",
"gal": "get a life",
"gf": "girlfriend",
"gfn": "gone for now",
"gmbo": "giggling my butt off",
"gmta": "great minds think alike",
"h8": "hate",
"hagn": "have a good night",
"hdop": "help delete online predators",
"hhis": "hanging head in shame",
"iac": "in any case",
"ianal": "I am not a lawyer",
"ic": "I see",
"idk": "I don't know",
"imao": "in my arrogant opinion",
"imnsho": "in my not so humble opinion",
"imo": "in my opinion",
"iow": "in other words",
"ipn": "I’m posting naked",
"irl": "in real life",
"jk": "just kidding",
"l8r": "later",
"ld": "later, dude",
"ldr": "long distance relationship",
"llta": "lots and lots of thunderous applause",
"lmao": "laugh my ass off",
"lmirl": "let's meet in real life",
"lol": "laugh out loud",
"ltr": "longterm relationship",
"lulab": "love you like a brother",
"lulas": "love you like a sister",
"luv": "love",
"m/f": "male or female",
"m8": "mate",
"milf": "mother I would like to fuck",
"oll": "online love",
"omg": "oh my god",
"otoh": "on the other hand",
"pir": "parent in room",
"ppl": "people",
"r": "are",
"rofl": "roll on the floor laughing",
"rpg": "role playing games",
"ru": "are you",
"shid": "slaps head in disgust",
"somy": "sick of me yet",
"sot": "short of time",
"thanx": "thanks",
"thx": "thanks",
"ttyl": "talk to you later",
"u": "you",
"ur": "you are",
"uw": "you’re welcome",
"wb": "welcome back",
"wfm": "works for me",
"wibni": "wouldn't it be nice if",
"wtf": "what the fuck",
"wtg": "way to go",
"wtgp": "want to go private",
"ym": "young man",
"gr8": "great"
}

def apos_short_dict(text, dictionary):
    """Consolidate apostrophies."""
    for word in text.split():
        if word.lower() in dictionary:
            if word.lower() in text.split():
                text = text.replace(word, dictionary[word.lower()])
    return text

def remove_html(txt):
    """Remove html."""
    txt = BeautifulSoup(txt, 'lxml')
    return txt.get_text()

def remove_punctuation(surveyText):
    """Remove any punctuation."""
    return "".join([i for i in surveyText if i not in string.punctuation])

def remove_stopwords(surveyText):
    """Remove stop words."""
    return [w for w in surveyText if w not in stopwordList]

def word_lemmatizer(surveyText):
    """Lemmatize tokens."""
    lemmatizer = WordNetLemmatizer()
    w_tokenizer = WhitespaceTokenizer()
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(surveyText)]

def word_stemmer(surveyText):
    """Stem words."""
    stemmer = PorterStemmer()
    return [stemmer.stem(i) for i in surveyText]

def fix_spelling(surveyText):
    sc = spellcorrector.SpellCorrector()
    return sc.correct_errors(surveyText)

def init_filter_dict(cleanedTxt):
    """Initialize id2txt dictionary to filter extremes."""
    dct = Dictionary(cleanedTxt.tolist())
    print(f'Total words before filter: {len(dct)}')
    dct.filter_extremes(no_below=2, no_above=0.15)
    print(f'Total words after filter: {len(dct)}')
    return dct

def extreme_filterer(surveyText, filter_dict):
    """Filter extremes"""
    docidx = filter_dict.doc2idx(surveyText)
    return np.array(surveyText)[np.array(docidx) != -1].tolist()


'''
    Control the parameter by putting value of TRUE or FALSE according to requirements.
    Args : txt - Provided text for preprocessing
            punctuation - Remove all punctuation, Initially value = False
            tokenize - Splitting long text into smaller lines
            stopwords - Remove such words which does not have much meaning to a line of text
            correct_apos - Remove apostrophe
            shortwords - Convert any short word to full meaningfull word
            specialCharacter - Replace all specialCharacter
            numbers - Remove numbers
            singleChar - Removing words whom length is one
            lematization - Lematize text
            stemming - Stemming any text
'''

def preprocessing(txt, punctuation= False, correct_spelling=False, tokenize= False, stopwords= False,
                  correct_apos= False, shortWords= False, specialCharacter= False, numbers= False, singleChar= False,
                  lematization= False, stemming= False, filter_extremes=False):

    cleanedTxt = txt.apply(lambda x: remove_html(x))

    if punctuation:
        cleanedTxt = cleanedTxt.apply(lambda x:remove_punctuation(x))

    if correct_spelling:
        cleaned_Txt = cleanedTxt.apply(lambda x: fix_spelling(x))

    if tokenize:
        cleanedTxt = cleanedTxt.apply(lambda x:word_tokenize(x.lower()))

    if stopwords:
        cleanedTxt = cleanedTxt.apply(lambda x: remove_stopwords(x))

    if correct_apos:
        cleanedTxt = cleanedTxt.apply(lambda x: apos_short_dict(str(x),apostrophe))

    if shortWords:
        cleanedTxt = cleanedTxt.apply(lambda x: apos_short_dict(str(x),short_words))

    if specialCharacter:
        '''Replacing Special Characters with space'''
        cleanedTxt = cleanedTxt.apply(lambda x: re.sub(r'[^a-zA-Z0-9]',' ',str(x)))

    if numbers:
        '''Replacing Numbers with space'''
        cleanedTxt = cleanedTxt.apply(lambda x: re.sub(r'[^a-zA-Z]',' ',x))

    if singleChar:
        '''Removing words whom length is one'''
        cleanedTxt = cleanedTxt.apply(lambda x: ' '.join([w for w in x.split() if len(w)>1]))

    if lematization:
        cleanedTxt = cleanedTxt.apply(lambda x: word_lemmatizer(x))

    if stemming:
        cleanedTxt = cleanedTxt.apply(lambda x: word_stemmer(x))

    if filter_extremes:
        filter_dict = init_filter_dict(cleanedTxt)
        cleanedTxt = cleanedTxt.apply(lambda x: extreme_filterer(x, filter_dict))

    return cleanedTxt


def build_embedding_space(data):
    '''
    Uses annoy nn approximation library to build an
    embedding space containing all words from the corpus
    and the mean embeddings of each cluster.

    Parameters
    ---------
        data : pd.DataFrame
            Dataframe containing all data, including
            mean comment embeddings and token ids.

    Returns
    ---------
        embedding_space : annoy
            Annoy nn tree.
    '''
    # Create the annoy embedding space.
    num_dimensions = data['embedding'].tolist()[0].size
    # Get embeddings for each token in the corpus.
    model = fasttext.load_model('topic_modeling\\crawl-300d-2M-subword.bin')
    # Get embeddings for each word for each comment.
    tok2emb = [(tok, model[tok]) for comment in data['comment_clean'] for tok in comment if len(comment) > 0]
    # Remove duplicate tokens.
    tok2emb = dict(tok2emb).items()

    # Add all tokens from the dataset.
    global index_to_token
    index_to_token = {}
    a = annoy.AnnoyIndex(num_dimensions, metric='angular')
    for i, (token, embedding) in enumerate(tok2emb):
        a.add_item(i, embedding)
        index_to_token[i] = token

    # Add cluster mean embeddings.
    global index_to_cluster
    global cluster_to_index
    index_to_cluster = {}
    cluster_to_index = {}
    # Continue numbering.
    j = i+1
    for cluster_id in sorted(list(data['cluster'].unique())):
        # filter incoming data by cluster_id
        cluster_data = data.loc[data['cluster'] == cluster_id]
        # calculate overall mean sentence embedding for cluster
        mean_embedding = np.mean(cluster_data['embedding'], axis=0)
        # add new entry based on mean embedding
        a.add_item(j, mean_embedding)
        index_to_cluster[j] = cluster_id
        cluster_to_index[cluster_id] = j
        j += 1

    a.build(n_trees=100)

    return a


def get_keywords(data, keywords, cluster_id):
    corpus = [' '.join(comment) for comment in data[data['cluster'] == cluster_id]['comment_clean'].tolist()]
    if keywords == 'tfidf':
        vectorizer = TfidfVectorizer(analyzer='word',
                 ngram_range=(1,1),
                 min_df=0.003,
                 max_df=0.5,
                 max_features=5000)
        scores = {}

        # Matrix of scored terms for each comment in the corpus.
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Mapping of terms to matrix ids.
        feature_names = vectorizer.get_feature_names()

        # For each comment extract the terms and scores and append them to a dictionary.
        for i in range(len(corpus)):
            feature_index = tfidf_matrix[i,:].nonzero()[1]
            tfidf_scores = zip([feature_names[y] for y in feature_index], [tfidf_matrix[i, x] for x in feature_index])
            scores.update(dict(tfidf_scores))

        return [key[0] for key in sorted(list(scores.items()), key=lambda x: x[1], reverse=True)]

    elif keywords == 'frequency':
        # create a general corpus of all tokenized comments
        corpus = []
        for comment_clean in data[data['cluster'] == cluster_id]['comment_clean']:
            # separate items in comments
            for item in comment_clean:
                corpus.append(item)
        # get frequency distribution of tokens
        freqdist = nltk.FreqDist(corpus)
        # get absolute number of most frequent word as normalizer
        norm = freqdist.most_common(5)

        return [item[0] for item in norm]


def get_sentences(data, cluster_id, method_sentences, n_sentences):
    '''
    :param data: dataframe with tokenized comments, cluster_ids, embeddings, and raw comments
    :param cluster_id: id of cluster representative sentences are needed for
    :param method_sentences: statistical (via word frequency) or embedding (central tendency of embeddings)
    :param n_sentences: number of desired representative sentences
    :return: tokenized comments of n most representative sentences
    '''

    if method_sentences == "statistical":
        # create a general corpus of all tokenized comments
        corpus = []
        for comment_clean in data[data['cluster'] == cluster_id]['comment_clean']:
            # separate items in comments
            for item in comment_clean:
                corpus.append(item)
        # get frequency distribution of tokens
        freqdist = nltk.FreqDist(corpus)
        # get absolute number of most frequent word as normalizer
        norm = freqdist.most_common(1)[0][1]
        # get relative frequency distribution by normalizing with most common token
        for key, value in freqdist.items():
            freqdist[key] = value / norm
        # calculate weighted frequency for each comment
        weightSentences = data[data['cluster'] == cluster_id][['comment_raw', 'comment_clean']]
        weights = []
        for comment_clean in weightSentences['comment_clean']:
            weightFreq = 0
            for item in comment_clean:
                weightFreq += freqdist[item]
            weights.append(weightFreq)
        weightSentences['weight'] = weights
        # sort list of weighted frequencies
        repr_sentences = pd.DataFrame.nlargest(weightSentences, columns='weight', n=n_sentences)
        # clear output of frequency weights etc
        return repr_sentences['comment_raw'].tolist()

    elif method_sentences == "embedding":
        # filter incoming data by cluster_id
        cluster_data = data.loc[data['cluster'] == cluster_id]
        # calculate overall mean sentence embedding for cluster
        mean_embedding = np.mean(cluster_data['embedding'], axis=0)
        # determine distance to mean for each embedding (cosine similarity)
        cluster_data['dist'] = cluster_data['embedding'].apply(lambda x: sp.spatial.distance.cosine(x, mean_embedding))
        # find n representative sentences with smallest distance
        repr_sentences = pd.DataFrame.nsmallest(cluster_data, columns='dist', n=n_sentences)
        # clean output
        return repr_sentences['comment_raw'].tolist()


def get_label(keywords, labels, cluster_id, embedding_space, max_cluster):
    if labels == 'top_5_words':
        return ' '.join(keywords[:5])

    elif labels == 'mean_projection':
        cluster_name_index = embedding_space.get_nns_by_item(cluster_to_index[cluster_id], n=max_cluster+1)
        for i in cluster_name_index:
            try:
                return index_to_token[i]
            except KeyError:
                continue


def export_graph(data, graph_path):
    # Visualize the reduced data with the cluster IDs.
    ax = sns.lmplot(data=data, x='PC1', y='PC2', hue='cluster',
                       fit_reg=False, legend=True, legend_out=True, scatter_kws={"s": 1})
    ax.savefig(graph_path)


def make_outputs_directory():
    '''
    Prepares an outputs directory for the current run on train/tune.

    Returns
    ---------
        output_path : str
            Filepath of the outputs directory.
    '''
    # Format output path with unique datetime identifier.
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_path = f'outputs\\output-{now}'
    # Create outputs directory.
    os.mkdir(output_path)

    return output_path


def evaluation(data, keywords, labels, method_sentences, n_sentences):
    print('Exporting results ...')
    output_path = make_outputs_directory()
    data_path = f'{output_path}\\data.csv'
    clusters_path = f'{output_path}\\clusters.csv'
    graph_path = f'{output_path}\\graph.png'

    embedding_space = build_embedding_space(data)
    data.to_csv(data_path)
    cluster_info = []
    max_cluster = max(list(data['cluster'].unique()))
    for cluster_id in sorted(list(data['cluster'].unique())):
        cluster_dict = {'cluster': cluster_id}
        cluster_dict['keywords'] = get_keywords(data, keywords, cluster_id)
        cluster_dict['label'] = get_label(cluster_dict['keywords'], labels, cluster_id, embedding_space, max_cluster)
        cluster_dict['sentences'] = get_sentences(data, cluster_id, method_sentences, n_sentences)
        cluster_info.append(cluster_dict)

    pd.DataFrame(cluster_info).to_csv(clusters_path)
    try:
        export_graph(data, graph_path)
    except KeyError as e:
        print('Unable to export graph: no dimensionality reduction.')

    return data_path, clusters_path, graph_path

