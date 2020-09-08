import copy
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
from nltk.corpus import stopwords


class TransSpell:
    """
    A context-sensitive spelling correction tool that uses pretrained transformer models to detect errors and supply
    corrections for them.
    """
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
        self.model = AutoModelWithLMHead.from_pretrained("distilbert-base-cased")
        self.stopwords = stopwords.words("english")

    def correct_errors(self, sequence: str) -> str:
        """
        Detects words that do not fit the context of the words surrounding it. If an error is detected, it is replaced
        by a word that better fits the context.

        :param sequence: The sentence that is to be checked for errors.
        :return: The input sequence, altered by replacing errors with their most likely suggestions.
        """
        temp_sent = sequence.split(" ")
        # if sequence only contains one word, return original sequence because approach does not work on one-word texts
        if len(temp_sent) == 1:
            return sequence
        replacement_token = self.tokenizer.mask_token
        for i, token in enumerate(sequence.split(" ")):
            # don't check stopwords and the first token of a sentence to reduce false positives
            if i == 0 or token in self.stopwords:
                continue
            sent = copy.deepcopy(temp_sent)
            sent[i] = replacement_token
            sent = " ".join(sent)
            results = self.generate_candidates(sent, topn=25)
            contained_in_suggestions = False
            for j, suggestion in enumerate(results):
                # decode suggestion
                suggestion = self.tokenizer.decode([suggestion])
                # overwrite code with string
                results[j] = suggestion
                if suggestion.lower() == token.lower():
                    contained_in_suggestions = True
                    break
            if not contained_in_suggestions:
                temp_sent[i] = results[0]
        return " ".join(temp_sent)

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

        return torch.topk(mask_token_logits, topn, dim=1).indices[0].tolist()


if __name__ == '__main__':
    ts = TransSpell()
    test_sents = ["We made ensure to meet the customer requirements in a consistent manner.",
                  "terst"]
    for sent in test_sents:
        results = ts.correct_errors(sent)
        print(results)
