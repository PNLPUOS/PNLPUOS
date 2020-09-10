import re
import copy
import collections
import torch
import enchant
import nltk
import time
import csv
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
from nltk.corpus import stopwords


class TransSpell:
    """
    A context-sensitive spelling correction tool that uses pretrained transformer models to detect errors and supply
    corrections for them.
    """
    def __init__(self, minimum_token_length: int = 3, maximum_frequency: int = 10, corpus_path: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
        self.model = AutoModelForMaskedLM.from_pretrained("distilbert-base-cased")
        self.stopwords = stopwords.words("english")
        self.char_minimum = minimum_token_length
        self.frequency_maximum = maximum_frequency
        self.dict_us = None
        self.dict_gb = None
        self.frequency_list = None
        if corpus_path:
            self.frequency_list = self.generate_frequency_list(corpus_path)

    def is_error(self, token: str) -> bool:
        """
        Uses a rule-based approach to evaluate whether a given token could be considered an error. This way of error
        detection is insensitive to context and will therefore only detect non-word errors.

        :param token: The word that is to be checked for being an error.
        :returns: True if token can be considered an error. False otherwise.
        """
        # load required components if they are not loaded yet
        if not self.dict_us:
            self.dict_us = enchant.Dict("en_US")
        if not self.dict_gb:
            self.dict_gb = enchant.Dict("en_GB")
        
        # step 1: check if token is long enough to be considered an error
        if len(token) <= self.char_minimum or re.match(r"^[A-Z]{,4}\W?s?$", token):
            return False

        # as capitalization is not relevant for the next steps, clean token for further processing
        token = self.clean_token(token)
        # if token consisted only of special characters that were removed, return false
        if not token:
            return False

        # step 2: check token frequency in corpus (if possible) to see if it is rare enough
        if self.frequency_list:
            if self.frequency_list[token] > self.frequency_maximum:
                return False

        # step 3: check whether token is contained in in English dictionaries (approximations of them)
        if self.dict_us.check(token) or self.dict_us.check(token.capitalize()) or \
            self.dict_gb.check(token) or self.dict_gb.check(token.capitalize()):
            return False

        return True

    def correct_errors(self, sequence: str) -> str:
        """
        Detects words that do not fit the context of the words surrounding it. If an error is detected, it is replaced
        by a word that better fits the context.

        :param sequence: The sentence that is to be checked for errors.
        :return: The input sequence, altered by replacing errors with their most likely suggestions.
        """
        # clean the sentence of excessive whitespaces; interpret them as commas
        sequence = re.sub("\s{2,}", ", ", sequence)
        orig_sent = sequence.split(" ")  # all changes made by correction will be stored in this list
        # if sequence is all-caps, make it all-lower as all-caps series cause issues with transformer
        original_all_caps = sequence.isupper()
        if original_all_caps:
            sequence = sequence.lower()
        temp_sent = sequence.split(" ")
        for i, token in enumerate(sequence.split(" ")):
            # don't check stopwords and the first token of a sentence to reduce false positives
            if i == 0 or i == len(sequence.split(" ")) or token in self.stopwords or \
               self.clean_token(token) in self.stopwords:
                continue
            # use a copy of the original sentence to avoid an incorrectly changed token to affect the intended context
            sent = copy.deepcopy(temp_sent)
            if self.is_error(token):
                sent[i] = self.tokenizer.mask_token  # mask token for which we want to generate suggestions
                sent = " ".join(sent)
                suggestions = self.generate_candidates(sent, topn=50)
                # if original token is contained in the list of results, consider it correct
                if token in suggestions:
                    continue
                correction = self.select_candidate(token, suggestions)
                if original_all_caps:
                    correction = correction.upper()
                orig_sent[i] = correction
                
        return " ".join(orig_sent)

    def select_candidate(self, original_token: str, candidates: list) -> str:
        """
        Determines the most likely replacement from a list of transformer-generated candidates given the original token.
        Prefers candidates with small Levenshtein distance to the original token and candidates with the same starting 
        letter as the original token.
        
        :param original_token: The word that is to be replaced.
        :param candidates: The list of candidates that could take the original_token's place given the sent context.
        :returns: The token from the candidate list that is most likely the best replacement.
        """        
        # dict for storing suggestions and their edit-distance
        candidate_edits = collections.defaultdict(list)
        candidate_ranking = []
        suggestion = None
        # sort candidates according to edit distance
        for candidate in candidates:
            candidate_edits[str(nltk.edit_distance(original_token, candidate))].append(candidate)
        # check for candidates within close edit distance
        for i in range(0, 4):
            if candidate_edits[str(i)]:
                temp_candidates = list(candidate_edits[str(i)])
                for candidate in temp_candidates:
                    candidate_ranking.append(candidate)
        for candidate in candidate_ranking:
            if candidate[0] == original_token[0]:
                suggestion = candidate
                break
        if suggestion:
            return suggestion
        # if no suggestion was found within edit distance, select the one deemed most likely by the model
        elif candidate_ranking:
            return candidate_ranking[0]
        else:
            return original_token

    def generate_candidates(self, sequence: str, topn: int = 5) -> list:
        """
        Given a sequence with one masked token, generate a list of likely candidates for that token based on the
        context of the surrounding words.

        :param sequence: The input sentence with one masked word.
        :param topn: The amount of candidates that is to be supplied.
        :return: A list of candidates that could be in the masked position.
        """
        input_str = self.tokenizer.encode(sequence, return_tensors="pt")
        mask_token_index = torch.where(input_str == self.tokenizer.mask_token_id)[1]
        token_logits = self.model(input_str)[0]
        mask_token_logits = token_logits[0, mask_token_index, :]
        results = torch.topk(mask_token_logits, topn, dim=1).indices[0].tolist()
        for j, suggestion in enumerate(results):
            # decode suggestion
            suggestion = self.tokenizer.decode([suggestion])
            # overwrite code with string
            results[j] = suggestion
        # return decoded candidates
        return results

    def generate_frequency_list(self, path: str) -> None:
        """
        Generate a Counter dict containing the frequency of every word in the corpus that spelling correction is to be 
        used on.

        :param path: The path to the file containing the texts that are to be analyzed.
        """
        # load data
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except FileNotFoundError:
            print("Corpus could not be found in provided path. Proceeding without frequency list.")
            return None
        self.frequency_list = collections.Counter()
        # tokenize each answer into a list of words
        for text in df["answers"]:
            answer = []
            for token in text.split():
                answer.append(self.clean_token(token))
            self.frequency_list.update(answer)

    def clean_token(self, token: str) -> str:
        """
        Cleans a given token by removing special characters and making them lowercase.

        :param token: The token that is to be cleaned.
        :returns: A cleaned version of the given token.
        """
        clean_token = ""
        for i, char in enumerate(list(token)):
            if re.match(r"\W", char):
                # if character is not one that might carry meaning for the word, do not add it to the clean token
                if char not in ["'", "-"]:
                    continue
                # if apostrophe is not in a valid position (i.e. the second to last token in the word), do not add it
                elif char == "'" and i != len(list(token)) - 2:
                    continue
            clean_token += char.lower()
        return clean_token


if __name__ == '__main__':
    start_time = time.time()
    ts = TransSpell(corpus_path="pnlp_data.csv")
    data = pd.read_csv("pnlp_data.csv", encoding="utf-8")
    sents = data["answers"]
    corrections = []
    for sent in tqdm(sents):
        corrections.append(ts.correct_errors(sent))
    print(f"Processed {len(corrections)} answers in {(time.time() - start_time)/60} minutes.")
    with open('results.csv', 'w', encoding="utf-8", newline='') as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["original", "correction"])
        for original, correction in zip(sents, corrections):
            if re.sub("\s{2,}", ", ", original) != correction:
                writer.writerow([original, correction])
