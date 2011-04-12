import luminoso
from luminoso.model import LuminosoModel
import tempfile, shutil
from nose.tools import assert_equal
import logging
logging.basicConfig()
from luminoso.whereami import get_project_filename
TEMPDIR = None

def setup_module():
    global TEMPDIR
    TEMPDIR = tempfile.mkdtemp()

def teardown_module():
    shutil.rmtree(TEMPDIR)

def test_english():
    english = luminoso.make_english(TEMPDIR+'/english')
    assert english.assoc.entry_named('cat', 'dog') > 0.5
    err1 = english.learn_assoc(1, 'foo', 'bar')
    assoc1 = english.assoc.entry_named('foo', 'bar')
    err2 = english.learn_assoc(1, 'foo', 'bar')
    assoc2 = english.assoc.entry_named('foo', 'bar')
    # after seeing the same example twice, error should decrease
    assert err2 < err1
    # after seeing the same example twice, association should increase
    assert assoc2 > assoc1

def test_small():
    model = LuminosoModel.make_empty(
        TEMPDIR + '/small',
        {
            'num_concepts': 3,
            'num_axes': 2,
            'iteration': 0,
            'reader': 'simplenlp.en'
        }
    )
    assert model.config['num_concepts'] == 3
    assert model.index_term('a', 2) == 0
    assert model.index_term('b', 1) == 1
    assert model.index_term('c', 3) == 2
    assert model.index_term('d', 4) == 1
    assert model.index_term('e', 0) == 0
    assert model.index_term('e', 0) == 0
    assert model.priority.items == ['e', 'd', 'c']

def test_read_from_url():
    model = LuminosoModel.make_empty(
        TEMPDIR + '/testdocs',
        {
            'num_concepts': 5,
            'num_axes': 2,
            'iteration': 0,
            'reader': 'simplenlp.en'
        }
    )
    model.learn_from_url(get_project_filename('test/TestDocuments'),
                         study=u'test')
    tags = model.database.get_document_tags(
      get_project_filename('test/TestDocuments/happytest.txt')
    )
    assert tags == [(u'study', u'test')]

if __name__ == '__main__':
    import cProfile
    import simplenlp
    en = simplenlp.get_nl('en')
    en.lemma_split('test')
    en.is_stopword('test')
    setup_module()
    model = LuminosoModel.make_empty(
        TEMPDIR + '/testdocs',
        {
            'num_concepts': 5,
            'num_axes': 2,
            'iteration': 0,
            'reader': 'simplenlp.en'
        }
    )
    cProfile.run('for i in xrange(10): model.learn_from_url("TestDocuments")', sort=2)
    #model = LuminosoModel('../models/PLDBStudy_test3')
    #cProfile.run("model.learn_from_url('../models/PLDBStudy/Documents')", sort=2)

    teardown_module()
