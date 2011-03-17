# -*- coding: utf-8 -*-
from simplenlp import get_nl

# TODO:
# - Handle situations that aren't just about discovering terms. (Negations,
#   different strengths of terms, etc.) Change the API to make this possible.

DOCUMENT = u'*Document*'

class TextReader(object):
    """
    An abstract class showing the interface that TextReaders must implement.
    """
    def extract_connections(self, text):
        """
        Outputs a list of tuples, containing two terms and the strength with
        which they are connected. These connections are symmetrical, so there
        is no need to output them once in each direction.
        
        It should also output a special connection at least once per term,
        with the first term being the special symbol called DOCUMENT.
        """
        raise NotImplementedError

class QuickEnglishTextReader(TextReader):
    """
    Uses simplenlp for handling English text.

    The model of connections between words is one that decreases geometrically
    until a cutoff value. The default values allow for associations between
    words that are up to 20 tokens apart.

    It also includes negative contexts. When one term in a pair appears in a
    negative context, the sign of the connection is flipped. The sign of the
    negated term's connection to the document is also flipped.
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

    def __init__(self, distance_weight=0.9, negation_weight=-0.5, cutoff=0.1):
        self.nl = get_nl('en')
        self.negation_weight = negation_weight
        self.distance_weight = distance_weight
        self.cutoff = cutoff

    def tokenize(self, text):
        return self.nl.tokenize_and_correct(text).split()
    
    def _attenuate(self, memory):
        for i in reversed(xrange(len(memory))):
            memory[i][1] *= self.distance_weight
            if memory[i][1] < self.cutoff:
                del memory[i]

    def extract_connections(self, text):
        c = self.__class__             # convenient shorthand
        
        tokens = self.tokenize(text)
        
        weight = 1.0
        memory = []
        prev_token = None

        for token in tokens:
            if token: # protect in case we get an empty token somehow
                self._attenuate(memory)
                if token in c.HARD_PUNCT:
                    memory = []
                    weight = 1.0
                    prev_token = None
                elif token in c.PUNCT_TOKENS or token[0] in c.PUNCT:
                    weight = 1.0
                    prev_token = None
                elif token in c.NEGATIONS:
                    weight *= self.negation_weight
                    prev_token = None
                elif (token.startswith('#') or token.startswith('+') or
                      token.startswith('-')):
                    # it's a tag
                    # FIXME: do some tag parsing.
                    if not token.startswith('#'):
                        token = '#' + token
                    yield (0, DOCUMENT, token)
                    prev_token = None
                else:
                    # this is an ordinary token, not a negation or punctuation
                    active_terms = [token]
                    if prev_token is not None:
                        bigram = prev_token + u' ' + token
                        active.append(bigram)
                    for term in active_terms:
                        yield (weight, DOCUMENT, term)
                        memory.append([term, weight])

                for term in active_terms:
                    for prev, prev_weight in memory:
                        # Currently, this will associate each term with
                        # itself, in addition to previous terms.
                        # Is this the right way to do that?
                        yield (weight*prev_weight, prev, term)

## Other things to include eventually:
# class JapaneseTextReader(TextReader):
#   (uses our CaboCha interface)
# 
# class ParsingEnglishTextReader(TextReader):
#   (takes constituents into account)
