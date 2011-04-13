"""
This file interacts with a SQLite file that keeps a running count of the
number of times various terms appear, and the number of times they appear
in each document.
"""
import math
from luminoso.text_readers import TAG

# all the messy SQLAlchemy imports
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.schema import Index
from sqlalchemy.sql.expression import desc
from sqlalchemy import create_engine, Column, Integer, Float, String, Text
try:
    import json
except ImportError:
    import simplejson as json
import logging
LOG = logging.getLogger(__name__)
#logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

def _expected_values(cont):
    """
    Expected values of a contingency table, from nltk.metrics.
    """
    n = float(sum(cont))
    results = []
    for i in range(4):
        results.append((cont[i] + cont[i ^ 1]) * (cont[i] + cont[i ^ 2]) / n)
    return results

_SMALL = 1e-22
_BIG = 1e9
def bigram_likelihood_ratio(n_12, n_1, n_2, n):
    """
    This function, borrowed from NLTK, calculates the significance of a bigram
    based on its unigram frequencies.

    - `n_12` is the number of occurrences of the entire bigram.
    - `n_1` is the number of occurrences of the first word.
    - `n_2` is the number of occurrences of the second word.
    - `n` is the total number of words learned.
    """
    n_o2 = n_2 - n_12
    n_1o = n_1 - n_12
    n_oo = n - n_12 - n_1o - n_o2
    contingency = [n_12+1, n_o2+1, n_1o+1, n_oo+1]
    expected = _expected_values(contingency)

    # the magic formula from NLTK
    likelihood = (2 * sum(obs * math.log(float(obs) / (exp + _SMALL) + _SMALL)
                         for obs, exp in zip(contingency, expected)))
    return likelihood

def json_encode(value):
    """
    Use JSON to represent any builtin value as a string.
    """
    return json.dumps(value)

ANY = '*'
Base = declarative_base()
class Term(Base):
    """
    Information about a term (a word or bigram), stored as a row in the
    database.

    These objects are not provided with knowledge about what database they are
    actually in, so the actual work must be done by :class:`TermDatabase`.
    """
    __tablename__ = 'terms'
    term = Column(String, primary_key=True)
    fulltext = Column(String, nullable=True)
    count = Column(Float, nullable=False)
    distinct_docs = Column(Integer, nullable=False)
    relevance = Column(Float, nullable=False, index=True)

    def __init__(self, term, count, distinct_docs, relevance):
        self.term = term
        self.count = count
        self.distinct_docs = distinct_docs
        self.relevance = relevance
    
    def __repr__(self):
        return "<%r, %d occurrences in %d documents>" % (self.term,
                                                         self.count,
                                                         self.distinct_docs)

class TermInDocument(Base):
    """
    Information about the number of occurrences of a term in a document.

    These objects are not provided with knowledge about what database they are
    actually in, so the actual work must be done by :class:`TermDatabase`.
    """
    __tablename__ = 'document_terms'
    id = Column(Integer, primary_key=True)
    term = Column(String, nullable=False, index=True)
    document = Column(String, nullable=False, index=True)
    count = Column(Integer, nullable=False)

    def __init__(self, term, document, count):
        self.term = term
        self.document = document
        self.count = count

    def __repr__(self):
        return "<%r in %r, %d occurrences>" % (self.term,
                                               self.document, 
                                               self.count)

Index('idx_term_document',
      TermInDocument.__table__.c.term,
      TermInDocument.__table__.c.document,
      unique=True)

class Feature(Base):
    """
    A table row storing some identified feature of a document that is not a
    term. As such, it should not be subject to TF-IDF.
    """
    __tablename__ = 'document_features'
    id = Column(Integer, primary_key=True)
    document = Column(String, nullable=False, index=True)
    key = Column(String, nullable=False, index=True)
    value = Column(String, nullable=False)  # JSON encoded

    def __init__(self, document, key, value):
        self.document = document
        self.key = key
        self.value = value
    
    def __repr__(self):
        return "<Feature on %r: %s=%s>" % (self.document, self.key, self.value)

class Document(Base):
    """
    A table row storing the text of a document. Contains the following fields:

    - id: a unique identifier for the document.
    - name: the human-readable document name.
    - reader: the process that extracted terms and features from the document.
    - text: a human-readable representation of the document.
    """
    __tablename__ = 'documents'
    id = Column(String, nullable=False, index=True, primary_key=True)
    name = Column(String, nullable=False)
    reader = Column(String, nullable=False)
    text = Column(Text, nullable=False)

    def __init__(self, docid, name, reader, text):
        self.id = docid
        self.name = name
        self.reader = reader
        self.text = text
    
    def __repr__(self):
        return "<Document %r: %r [%s]>" % (self.id, self.name, self.reader)

#class GlobalData(Base):
#    __tablename__ = 'global_data'
#    key = Column(String, primary_key=True)
#    value = Column(Integer)
#
#    def __init__(self, key, value):
#        self.key = key
#        self.value = value

class TermDatabase(object):
    """
    A SQLite database that counts terms and their occurrences in documents.
    """
    def __init__(self, filename):
        """
        Open or create a TermDatabase, given its filename.
        """
        # this boilerplate is turning me off from sqlalchemy. Why can't I have
        # one object that I interact with, instead of four levels of crap?
        self.sql_engine = create_engine('sqlite:///'+filename)
        self.sql_session_maker = sessionmaker(bind=self.sql_engine)
        self.sql_session = self.sql_session_maker()
        Base.metadata.create_all(bind=self.sql_engine)
        self.term_count_cache = {}

    def _increment_term_count(self, term, value=1, newdoc=False):
        """
        Increment the recorded number of times we have seen `term` by
        `value`. Also, if `newdoc` is True, increment the number of distinct
        documents containing the term.
        """
        term_entry = self.sql_session.query(Term).get(term)
        if term_entry:
            term_entry.count += value
            if newdoc:
                term_entry.distinct_docs += 1
            self.term_count_cache[term] = term_entry.count
        else:
            assert newdoc, "Something is wrong -- how did this term "\
                           "appear without appearing in a document?"
            term_entry = Term(term, value, 1, 0)
            self.sql_session.add(term_entry)
        
    def _increment_term_document_count(self, document, term, value=1):
        """
        Increments the count of a term in a document. Returns True if that
        term has not appeared in that document before, so that the Term
        entry can also make note of that.
        """
        query = self.sql_session.query(TermInDocument)\
                  .filter(TermInDocument.term == term)\
                  .filter(TermInDocument.document == document)
        try:
            term_entry = query.one()
            term_entry.count += value
            return False
        except NoResultFound:
            term_entry = TermInDocument(term, document, value)
            self.sql_session.add(term_entry)
            return True
    
    def increment_term_in_document(self, document, term, value=1):
        """
        Record the fact that a given term appeared in a given document, with
        weight `value`. This will additionally update the relevance value
        of the term, and if the term is a tag, record the tag on the document.

        This does not automatically commit changes to the database.
        """
        if isinstance(term, tuple) and term[0] == TAG:
            return self.set_tag_on_document(document, term[1], term[2])
        newdoc = self._increment_term_document_count(document, term, value)
        absv = abs(value)
        self._increment_term_count(term, absv, newdoc)
        newdoc_any = self._increment_term_document_count(document, ANY, absv)
        self._increment_term_count(ANY, absv, newdoc_any)
        self._update_term_relevance(term)
    
    def set_tag_on_document(self, document, key, value):
        """
        Record the fact that this document has this tag as a Feature in the
        database.

        This does not automatically commit changes to the database.
        """
        query = self.sql_session.query(Feature)\
                  .filter(Feature.document == document)\
                  .filter(Feature.key == key)
        try:
            tag_entry = query.one()
            tag_entry.value = json_encode(value)
        except NoResultFound:
            tag_entry = Feature(document, key, json_encode(value))
            self.sql_session.add(tag_entry)
    
    def get_term_text(self, term):
        term_entry = self.sql_session.query(Term).get(term)
        if term_entry:
            return term_entry.fulltext or term
        else:
            return term        

    def set_term_text(self, term, fulltext):
        """
        After observing the use of a term in a document, record the full text
        that it came from. If the term is not in the database, this has no
        effect.

        This does not automatically commit changes to the database.
        """
        term_entry = self.sql_session.query(Term).get(term)
        if term_entry:
            term_entry.fulltext = fulltext

    # removed find_term_texts; this is an inefficient place to do that.
    
    def add_document(self, doc_info):
        """
        Record the terms in a document in the database. If the database already
        has a document with this name, that document will be replaced.

        The terms must already be extracted by some other process, and saved
        into the document dictionary as the key 'terms'.
        `reader_name` indicates which reader was used.
        """
        docname = doc_info[u'name']
        docid = doc_info[u'url']
        terms = doc_info[u'terms']
        text = doc_info[u'text']
        reader_name = doc_info[u'reader']
        doc = self.sql_session.query(Document).get(docid)
        if doc is not None:
            if doc.text == text and doc.reader == reader_name:
                # nothing has changed, so return
                return
            self._clear_document(docid)
        
        for term in terms:
            if isinstance(term, tuple):
                # this is a (term, value) tuple
                term, value = term
            else:
                value = 1
            self.increment_term_in_document(docid, term, value)

        for key, value in doc_info.get('tags', []):
            self.set_tag_on_document(docid, key, value)
        
        doc = Document(docid, docname, reader_name, text)
        self.sql_session.add(doc)
        self.commit()

    def get_document(self, docid):
        """
        Get a Document from the database by its ID (URL).
        """
        return self.sql_session.query(Document).get(docid)
    
    def get_document_tags(self, docid):
        """
        Get a list of (key, value) pairs representing all the tags on this
        document.
        """
        return [(key, json.loads(value))
                for key, value
                in self.sql_session.query(Feature)
                       .filter(Feature.document == docid)
                       .values(Feature.key, Feature.value)]
    
    def _clear_document(self, document):
        """
        Remove the information that we got from a given Document, given its
        name.
        """
        query = self.sql_session.query(TermInDocument)\
                    .filter(TermInDocument.document == document)
        doc = self.sql_session.query(Document).get(document)
        for row in query.all():
            term, document, count = row
            term_entry = self.sql_session.query(Term).get(term)
            term_entry.count -= count
            term_entry.distinct_docs -= 1
            row.delete()
        any_term = self.sql_session.query(Term).get(ANY)
        any_term.distinct_docs -= 1
        doc.delete()
    
    def clear_document(self, document):
        """
        Remove the information that we got from a given Document, given its
        name.
        """
        self._clear_document(document)
        self.commit()
    
    def count_term(self, term):
        """
        Returns the number of times we have seen this term.
        From the cache, if possible.
        """
        if term in self.term_count_cache:
            return self.term_count_cache[term]
        term_entry = self.sql_session.query(Term).get(term)
        if term_entry:
            self.term_count_cache[term_entry] = term_entry.count
            return term_entry.count
        else:
            self.term_count_cache[term_entry] = 0
            return 0

    def term_relevance(self, term):
        """
        Returns the relevance of the term, which is either its unigram
        frequency or its bigram likelihood ratio (a function defined by NLTK
        that expresses when a bigram occurs more often than its unigram parts
        would imply).
        """
        words = term.split(' ')
        if isinstance(term, tuple):
            # probably a tag
            return _BIG
        if len(words) == 1:
            return self.count_term(term)
        elif len(words) == 2:
            return bigram_likelihood_ratio(
                self.count_term(term),
                self.count_term(words[0]),
                self.count_term(words[1]),
                self.count_term(ANY)
            )
        else:
            raise NotImplementedError(
                "I don't know how to handle trigrams or larger"
            )
    
    def top_terms(self, nterms):
        """
        Get the `nterms` most relevant terms.
        """
        return self.sql_session.query(Term)\
                   .order_by(desc(Term.relevance))[:nterms]
    
    def _update_term_relevance(self, term):
        """
        Calculate the new relevance value of a term and store it in the
        database (but don't commit yet).
        """
        #self.sql_engine.execute(
        #  "update terms set relevance=:relevance where term=:term",
        #  term=term, relevance=self.term_relevance(term)
        #)
        term_entry = self.sql_session.query(Term).get(term)
        term_entry.relevance = self.term_relevance(term)

    def count_term_in_document(self, term, document):
        """
        Get the number of times the given term appeared in the given document.
        """
        query = self.sql_session.query(TermInDocument)\
                  .filter(TermInDocument.term == term)\
                  .filter(TermInDocument.document == document)
        try:
            term_entry = query.one()
            return term_entry.count
        except NoResultFound:
            return 0
    
    def count_term_distinct_documents(self, term):
        """
        Get the number of distinct documents a term has appeared in.
        """
        term_entry = self.sql_session.query(Term).get(term)
        if term_entry:
            return term_entry.distinct_docs
        else:
            return 0

    def count_documents(self):
        """
        Get the total number of documents we have seen.
        """
        return self.count_term_distinct_documents(ANY)
    
    def all_documents(self):
        """
        Get a list of all the document IDs stored in the database.
        """
        return [item[0] for item in
                self.sql_session.query(Document).values(Document.id)]
    list_documents = all_documents
    
    #def set_global(self, key, value):
    #    global_entry = self.sql_session.query(GlobalData).get(key)
    #    if global_entry:
    #        global_entry.value = value
    #    else:
    #        global_entry = GlobalData(key, value)
    #        self.sql_session.add(global_entry)
    #
    #def get_global(self, key):
    #    global_entry = self.sql_session.query(GlobalData).get(key)
    #    if global_entry:
    #        return global_entry.value
    #    else:
    #        return None
    #
    #def increment_global(self, key, value):
    #    global_entry = self.sql_session.query(GlobalData).get(key)
    #    if global_entry:
    #        global_entry.value += 1
    #    else:
    #        global_entry = GlobalData(key, 1)
    #        self.sql_session.add(global_entry)

    def term_idf(self, term):
        """
        Get the IDF value for a given term.
        """
        idf = math.log(2 + self.count_term_distinct_documents(ANY))\
            - math.log(1 + self.count_term_distinct_documents(term))
        return idf        

    def tfidf_term_in_document(self, term, document):
        """
        Calculate the TF-IDF (Term Frequency by Inverse Document Frequency)
        value for a given term in a given document.
        """
        tf = self.count_term_in_document(term, document)\
           / self.count_term_in_document(ANY, document)
        idf = math.log(1 + self.count_term_distinct_documents(ANY))\
            - math.log(1 + self.count_term_distinct_documents(term))
        return tf * idf
    
    def all_tags(self):
        """
        Get a list of all (key, value) pairs that define tags on this dataset.
        """
        tags = set()
        query = self.sql_session.query(Feature).all()
        for tag in query:
            tags.add((tag.key, json.loads(tag.value)))
        return tags

    def documents_with_tag(self, tag):
        """
        Get a generator of document IDs with a certain tag present, along with
        the value of that tag.

        TODO: test
        """
        query = self.sql_session.query(Feature)\
                    .filter(Feature.key == tag)
        for row in query:
            yield row.document, json.loads(row.value)
    
    def documents_with_tag_value(self, tag, value):
        """
        Find documents where a certain tag has a certain value.
        Returns a generator of document IDs.

        TODO: test
        """
        jvalue = json.dumps(value)
        query = self.sql_session.query(Feature)\
                    .filter(Feature.key == tag).filter(Feature.value == jvalue)
        for row in query:
            yield row.document

    def commit(self):
        """
        Commit all changes to the database.
        """
        self.sql_session.commit()

