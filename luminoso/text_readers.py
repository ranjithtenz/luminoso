# -*- coding: utf-8 -*-
"""
This module defines a variety of classes that can be used to read plain text
and return word associations. Which reader to use depends on the complexity
of the task and what language the text is in.
"""
from simplenlp import get_nl

# TODO:
# - Handle situations that aren't just about discovering terms. (Negations,
#   different strengths of terms, etc.) Change the API to make this possible.

DOCUMENT = u'*Document*'
TAG = u'*Tag*'

class TextReader(object):
    """
    An abstract class showing the interface that TextReaders must implement.
    """
    def __init__(self):
        raise NotImplementedError("TextReader is an abstract class")

    def extract_connections(self, text):
        """
        Outputs a list of tuples, containing two terms and the strength with
        which they are connected. These connections are symmetrical, so there
        is no need to output them once in each direction.
        
        It should also output a special connection at least once per term,
        with the first term being the special symbol called DOCUMENT.
        """
        raise NotImplementedError

class SimpleNLPEnglishReader(TextReader):
    """
    Uses simplenlp for handling English text.

    The model of connections between words is one that decreases geometrically
    until a cutoff value. The default values allow for associations between
    words that are up to 20 tokens apart.

    It also includes negative contexts. When one term in a pair appears in a
    negative context, the sign of the connection is flipped. The sign of the
    negated term's connection to the document is also flipped.

    TODO: factor out common things into SimpleNLPReader.
    """
    # this punctuation symbol marks the strongest possible division in a text,
    # so that no word associations can occur across it. We intend for this
    # to be inserted manually. For example, if two reviews from different
    # people got concatenated, write "text of first review // text of second".
    HARD_PUNCT = [u'//']

    # punctuation characters that end a negative context; tokens that begin
    # with one of these such as "..." or "?!" count as well.
    PUNCT = [u'.', u'?', u'!', u':', u'…', u'-', u'—']

    # punctuation that has to appear as the entire token to count
    # (so we don't accidentally treat "'s" as punctuation, for example)
    PUNCT_TOKENS = [u"''", u"``", u"'", u"`", u":", u";"]

    # words that flips the context from positive to negative
    NEGATIONS = [u'no', u'not', u'never', u'stop', u'lack',
                 u"n't", u'without']
    
    EXTRA_STOPWORDS = [
        'also', 'not', 'without', 'ever', 'because', 'then', 
        'than', 'do', 'just', 'how', 'out', 'much', 'both', 'other'
    ]

    def __init__(self, distance_weight=0.9, negation_weight=-0.5, cutoff=0.1):
        """
        Create a reader that reads English text using `simplenlp`. You can
        optionally adjust the weights for how much various terms affect
        each other.
        """
        self.nl = get_nl('en')
        self.negation_weight = negation_weight
        self.distance_weight = distance_weight
        self.cutoff = cutoff

    def tokenize(self, text):
        """
        Return a list of the (lemmatized, non-stopword) tokens that appear in
        the text.
        """
        spaced = self.nl.lemma_split(text)[0]
        return spaced.split()
    
    def _attenuate(self, memory):
        """
        Decrease the weights of the terms that are currently in memory, as
        they get farther from our place in the text.
        Remove them if they fall below the given cutoff value.
        """
        for i in reversed(xrange(len(memory))):
            memory[i][1] *= self.distance_weight
            if memory[i][1] < self.cutoff:
                del memory[i]

    def extract_connections(self, text):
        """
        Determine which terms appear near each other in the given document,
        and output a generator of (weight, term1, term2) connections that
        can be used to train a LuminosoSpace.

        The special term named `DOCUMENT` is used to mark the strength with
        which each term appears in the document itself.
        """
        cls = self.__class__             # convenient shorthand
        
        tokens = self.tokenize(text)
        
        weight = 1.0
        memory = []
        prev_token = None

        for token in tokens:
            if token: # protect in case we get an empty token somehow
                self._attenuate(memory)
                active_terms = []
                if token in cls.HARD_PUNCT:
                    memory = []
                    weight = 1.0
                    prev_token = None
                elif token in cls.PUNCT_TOKENS or token[0] in cls.PUNCT:
                    weight = 1.0
                    prev_token = None
                elif token in cls.NEGATIONS:
                    weight *= self.negation_weight
                    prev_token = None
                elif token in cls.EXTRA_STOPWORDS:
                    continue
                elif (token.startswith(u'#') or token.startswith(u'+') or
                      token.startswith(u'-')) and len(token) > 2:
                    # it's a tag
                    tag = parse_tag(token)
                    yield (0, DOCUMENT, tag)
                    prev_token = None
                else:
                    # this is an ordinary token, not a negation or punctuation
                    active_terms = [token]
                    if prev_token is not None:
                        bigram = prev_token + u' ' + token
                        active_terms.append(bigram)
                    for term in active_terms:
                        yield (weight, DOCUMENT, term)
                        memory.append([term, weight])

                for term in active_terms:
                    for prev, prev_weight in memory:
                        # Currently, this will associate each term with
                        # itself, in addition to previous terms.
                        # Is this the right way to do that?
                        yield (weight*prev_weight, prev, term)
    
def parse_tag(token):
    """
    If this token is a specially-marked tag, parse it into a consistent format.

    Returns a tuple of `(TAG, key, value)`, where `TAG` is a special flag
    value, `key` is the name of the tag, and `value` is the value it is
    assigned (or None).
    """
    if token.startswith(u'#'):
        if u'=' in token:
            key, value = token.split(u'=', 1)
            # handle numeric values
            try:
                value = float(value)
            except ValueError:
                pass
        else:
            key = token[1:]
            value = None
    elif token.startswith(u'+'):
        key = token[1:]
        value = True
    elif token.startswith(u'-'):
        key = token[1:]
        value = False
    else:
        # this shouldn't happen, but if it does, it's not worth throwing
        # an error.
        key = token
        value = None
    return (TAG, key, value)

READERS = {
    'simplenlp.en': SimpleNLPEnglishReader,
}

def get_reader(name):
    """
    Gets a TextReader instance by specifying its name. Often, the reader
    you want will be 'simplenlp.en'.
    """
    try:
        return READERS[name]()
    except KeyError:
        raise KeyError("There is no text reader named %r." % name)

## Other things to include eventually:
# class JapaneseTextReader(TextReader):
#   (uses our CaboCha interface)
# 
# class ParsingEnglishTextReader(TextReader):
#   (takes constituents into account)
