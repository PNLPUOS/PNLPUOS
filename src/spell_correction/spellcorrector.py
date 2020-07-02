from spellchecker import SpellChecker
from nltk.tokenize import sent_tokenize, word_tokenize


class SpellCorrector:
    """
    Object that is capable of recognizing spelling mistakes and suggesting a more correct
    alternative candidate.
    Currently uses pyspellchecker library to make spelling correction accessible in the pipeline.
    Will be replaced by more sophisticated solution as it's furthering in development.
    """

    def __init__(self):
        self.sc = SpellChecker()

    def correct_errors(self, text):
        """
        Takes an uncorrected text and applies spelling correction to it. Returns corrected text.
        """
        corrected_sents = [self.correct_sentence(sent) for sent in sent_tokenize(text)]
        return corrected_sents

    def correct_sentence(self, sent):
        """
        Takes an uncorrected sentence and applies spelling correction to it. Returns corrected sentence.
        """
        corrected_words = [self.correct_word(word) for word in word_tokenize(sent)]
        return corrected_words

    def correct_word(self, word):
        """
        Takes an uncorrected word and applies spelling correction to it. Returns corrected word.
        """
        return self.sc.correction(word)


if __name__ == '__main__':
    txt = "This is a test txet. It hsa a fe typo's."
    sc = SpellCorrector()
    print(sc.correct_errors(txt))
