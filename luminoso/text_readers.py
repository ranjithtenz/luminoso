class TextReader(object):
    def tokenize(self, text):
        raise NotImplementedError
    
    def split_sentences(self, text):
        raise NotImplementedError

    def extract_terms_by_sentence(self, text):
        raise NotImplementedError
    
    def extract_terms(self, text):
        raise NotImplementedError