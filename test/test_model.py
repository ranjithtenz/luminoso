import luminoso
from luminoso.model import LuminosoModel
import tempfile, shutil
from nose.tools import assert_equal

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
            'num_concepts': 5,
            'num_axes': 2,
            'iteration': 0,
            'reader': 'simplenlp.en'
        }
    )
    assert model.config['num_concepts'] == 5

