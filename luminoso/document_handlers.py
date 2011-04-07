"""
A document handler is the step before a TextReader: it takes in some source of
raw data, and yields dictionaries representing individual documents. A
TextReader can then scan these for terms and tags.
"""
import chardet
import codecs
import os
try:
    import json
except ImportError:
    import simplejson as json
import logging
LOG = logging.getLogger(__name__)

def ensure_unicode(text):
    """
    Take in something. Make it Unicode.
    """
    if isinstance(text, unicode):
        return text
    elif isinstance(text, str):
        return unicode(text, errors='replace')
    else:
        return unicode(text)

def handle_text_file(filename, name=None):
    """
    Handle a file that we believe to contain plain text, in some reasonable
    encoding.
    """
    # open the raw text first, to determine its encoding
    rawtext = open(filename, 'rb')
    encoding = chardet.detect(rawtext.read(1024))['encoding']
    rawtext.close()

    text = codecs.open(filename, encoding=encoding, errors='replace').read()
    for result in handle_text(text, filename, name):
        yield result

def handle_text(text, url, name=None):
    """
    Given plain text content, return it as a document dictionary.
    """
    name = ensure_unicode(name or os.path.basename(url))
    yield {u'name': name, u'url': url, u'text': ensure_unicode(text)}

def _check_document(document):
    """
    Upon retrieving a document dictionary, see if it fits the format Luminoso
    expects.
    """
    return (isinstance(document, dict) and 
            u'name' in document and u'text' in document)

def handle_json_file(filename, name=None):
    """
    Dig into a JSON file, and find documents containing "name" and "text"
    entries.
    """
    stream = open(filename)
    obj = json.load(stream)
    stream.close()
    for result in handle_json_obj(obj, filename, name):
        yield result

def handle_json_obj(obj, url, name=None):
    """
    Handle a JSON object, which is either a document itself or may contain
    a number of subdocuments.

    The `url` parameter should contain a filename or URL to ensure that
    document names are not completely ambiguous.
    """
    # TODO: split these cases into separate functions that are reusable
    # (pylint has a valid complaint!)
    if isinstance(obj, basestring):
        fullname = url
        if name:
            fullname = url + u'#' + name
        doc = {
            u'url': fullname,
            u'name': name,
            u'text': obj
        }
        yield doc
    elif isinstance(obj, list):
        for document in obj:
            # discard the name; it won't be unique
            for result in handle_json_obj(document, url):
                yield result
    elif isinstance(obj, dict):
        baseurl = os.path.dirname(url)
        if u'text' in obj:
            # this is a single document on its own
            obj[u'url'] = url + u'#' + obj.get(u'name', name)
            obj[u'name'] = obj.get(u'name', name)
            yield obj
        elif u'url' in obj:
            for result in handle_url(baseurl + os.path.sep + obj[u'url'],
                                     obj.get(u'name', name)):
                yield result
        else:
            # assume it's a dictionary mapping name -> document
            for newname, document in obj.items():
                for result in handle_json_obj(document, url, newname):
                    yield result
    else:
        LOG.warn("could not find a valid document in %r#%r" % (url, name))

def handle_directory(dirname):
    """
    Handle a directory and get a stream of documents out of it.
    """
    for file in os.listdir(dirname):
        if not file.startswith('.'):
            for result in handle_url(dirname+os.sep+file):
                yield result

def handle_url(url, name=None):
    """
    Handle a file specified by its URL (by default, a local file).

    TODO: handle schemas that aren't local files.
    """
    if os.access(url, os.F_OK):
        if os.access(url+u'/', os.F_OK): # it's a directory
            for result in handle_directory(url):
                yield result
        elif url.endswith(u'.json') or url.endswith(u'.js'):
            for result in handle_json_file(url):
                yield result
        else:
            # assume text file
            for result in handle_text_file(url):
                yield result
    else:
        if name:
            LOG.warn("could not open %r#%r" % (url, name))
        else:
            LOG.warn("could not open %r" % url)
