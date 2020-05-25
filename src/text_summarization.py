"""
module providing text summarization pipeline
using huggingface transformers and pytorch
"""
import pandas as pd
from typing import List
import spacy
import torch
from transformers import BertTokenizer, BertModel
from transformers import pipeline
from transformers.pipelines import SummarizationPipeline


def prepare_data(data: pd.Series) -> List[str]:
    """
    perform following preparation steps:
    - split comments to sentences
    - tokenize for BERT input
    - combine tokens to one text
    - limit number of tokens to 512
    (BERT restriction)
    :param data:
    :return:
    """

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized_text = []
    # split to sentences
    nlp = spacy.blank("en")
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)

    for i, comment in enumerate(data):
        doc = nlp(comment)
        for sentence in doc.sents:
            encoded_sentence = tokenizer.encode(sentence.text)
            tokenized_text += encoded_sentence
        if i == 10:
            break

    return [tokenized_text]


def text_summary_pipeline(data: pd.Series):
    """
    main pipeline which orchestrates the steps
    for the text summary
    :param data:
    :return:
    """

    summarizer = pipeline("summarization")

    # prepare data for BERT
    processed_data = prepare_data(data)
    for text in processed_data:
        output = summarizer(text)
        print(output)


def main():
    """
    just a dummy main to avoid running
    the whole pipeline everytime
    :return:
    """
    data_path = '../data/pnlp_data_en.csv'
    data = pd.read_csv(
        data_path,
        delimiter=';'
    )['Comments']

    # actually running the text summary pipeline
    text_summary_pipeline(data)



if __name__ == '__main__':
    main()