from luminoso.document_handlers import handle_url
from luminoso.whereami import get_project_filename
import nose


def test_directory():
    test_dir = get_project_filename('test/TestDocuments')
    good = [{u'name': u'happytest.txt',
             u'text': u'i am happy and not sad\n',
             u'url': test_dir + u'/happytest.txt'},
            {u'name': u'Entry 1',
             u'text': u'one two three',
             u'url': test_dir + u'/list.json#Entry 1'},
            {u'name': u'Entry 2',
             u'text': u'two three four',
             u'url': test_dir + u'/list.json#Entry 2'},
            {u'name': u'relative.txt',
             u'text': u'three four five\n',
             u'url': test_dir + u'/../MoreTestDocuments/relative.txt'},
            {u'name': u'JSON test',
             u'text': u'one two three',
             u'url': test_dir + u'/simple.json#JSON test'},
            {u'name': u'unicode test',
             u'text': u'\u4e00\u4e8c\u4e09\u56db\u4e94',
             u'url': test_dir + u'/unicode.json#unicode test'},
            {u'name': u'unicode.txt',
             u'text': u'\u4e94\u56db\u4e09\u4e8c\u4e00\n',
             u'url': test_dir + u'/unicode.txt'}]
    got = handle_url(get_project_filename('test/TestDocuments'))
    nose.tools.assert_equal(sorted(list(good)), sorted(list(got)))