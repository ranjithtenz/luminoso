from luminoso.model import LuminosoModel
import tempfile, shutil

TEMPDIR = None

def setup_module():
    global TEMPDIR
    TEMPDIR = tempfile.mkdtemp()

def teardown_module():
    shutil.rmtree(TEMPDIR)

def test_documents():
    themodel = LuminosoModel.make_empty(TEMPDIR+'/test_documents')
    #themodel.add_document(dict(url='#test1', name='test one',
    #                           text='one two three'))
    themodel.add_document(dict(url='#test1', name='test one',
                               text='one two three'))
    themodel.add_document(dict(url='#test2', name='test two',
                               text='two three four'))
    assert themodel.database.count_documents() == 2
