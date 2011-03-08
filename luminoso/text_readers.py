from simplenlp import get_nl

class TextReader(object):
    """
    An abstract class showing the interface that TextReaders must implement.
    """
    def tokenize(self, text):
        """
        Split a text into words.
        """
        raise NotImplementedError
    
    def split_sentences(self, text):
        """
        Split a text into sentences.
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
        """
        raise NotImplementedError

class QuickEnglishTextReader(TextReader):
    """
    Uses simplenlp for handling English text.
    """
    def __init__(self):
        self.nl = get_nl('en')

    def tokenize(self, text):
        return self.nl.tokenize(text).split()
    
    # TODO: finish

## Other things to include eventually:
# class JapaneseTextReader(TextReader):
#   (uses our CaboCha interface)
# 
# class ParsingEnglishTextReader(TextReader):
#   (takes constituents into account)
