from luminoso.text_readers import get_reader
from nose.tools import assert_equal, raises

def test_simple():
    reader = get_reader('simplenlp.en')
    tokenized = list(reader.extract_tokens('one two three four'))
    assert_equal(tokenized, ['two', 'three', 'four'])

def test_tagged():
    reader = get_reader('simplenlp.en')
    tokenized = list(reader.extract_tokens('one two three #four'))
    assert_equal(tokenized, ['two', 'three', '#four'])

@raises(KeyError)
def test_no_reader():
    get_reader('simplenlp.foo')