# -*- coding: utf-8 -*-
from simplenlp import get_nl

# TODO:
# - Handle situations that aren't just about discovering terms. (Negations,
#   different strengths of terms, etc.) Change the API to make this possible.


class TextReader(object):
    """
    An abstract class showing the interface that TextReaders must implement.
    """
    def tokenize(self, text):
        """
        Split a text into words.
        """
        raise NotImplementedError

    def extract_terms_by_sentence(self, text):
        """
        Extract all terms from a text (including multi-word terms), returning
        a list of lists with one list per sentence.
        """
        raise NotImplementedError
    
    def extract_terms(self, text):
        """
        Extract all terms from a text (including multi-word terms), ignoring
        sentence boundaries if they are present.

        The default implementation of this will simply flatten the results
        of extract_terms_by_sentence.
        """
        return [word for sentence in self.extract_terms_by_sentence(text)
                     for word in sentence]

class QuickEnglishTextReader(TextReader):
    """
    Uses simplenlp for handling English text.
    """
    # punctuation that marks a high-level division in a text; tokens that begin
    # with one of these such as "..." or "?!" count as well
    SENTENCE_PUNCT = [u'.', u'?', u'!', u':', u'…']

    # punctuation characters that should end a negative context
    PUNCT = SENTENCE_PUNCT + [u'-', u':', u';', u'—']

    # punctuation that have to appear as the entire token to count
    # (so we don't accidentally treat "'s" as punctuation)
    PUNCT_TOKENS = [u"''", u"``", u"'", u"`"]

    # words that flips the context from positive to negative
    NEGATIONS = [u'no', u'not', u'never', u'stop', u'lack',
                 u"n't", u'without']

    def __init__(self):
        self.nl = get_nl('en')

    def tokenize(self, text):
        return self.nl.tokenize(text).split()
    
    def extract_terms_by_sentence(self, text):
        sentences = []
        tokens = self.tokenize(text)
        current_sentence = []
        for token in tokens:
            if token: # protect against empty strings if they show up somehow
                if token[0] in PUNCT or token in FIDDLY_PUNCT:


    # TODO: finish

## Other things to include eventually:
# class JapaneseTextReader(TextReader):
#   (uses our CaboCha interface)
# 
# class ParsingEnglishTextReader(TextReader):
#   (takes constituents into account)
