"""
A DocumentHandler is the step before a TextReader: it takes in some source of
raw data, and yields dictionaries representing individual documents. A
TextReader can then scan these for terms and tags.
"""
import chardet
import codecs

def handle_text_file(filename):
    """
    Handle a file that we believe to contain plain text, in some reasonable
    encoding.
    """
    # open the raw text first, to determine its encoding
    rawtext = open(filename, 'rb')
    encoding = chardet.detect(rawtext.read())['encoding']
    rawtext.close()

    text = codecs.open(filename, encoding=encoding, errors='replace').read()
    filename = unicode(filename, errors='replace')
    return dict(name=filename, text=text)

