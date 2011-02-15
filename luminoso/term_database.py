"""
This file interacts with a SQLite file that keeps a running count of the
number of times various terms appear, and the number of times they appear
in each document.
"""
import math

# all the messy SQLAlchemy imports
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.schema import Index
from sqlalchemy import create_engine, Column, Integer, Float, String, Text

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
    contingency = [n_12, n_o2, n_1o, n_oo]
    expected = _expected_values(contingency)

    # the magic formula from NLTK
    likelihood = (2 * sum(obs * math.log(float(obs) / (exp + _SMALL) + _SMALL)
                         for obs, exp in zip(contingency, expected)))
    return likelihood

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
    count = Column(Integer, nullable=False)
    words = Column(Integer, nullable=False)
    relevance = Column(Float, nullable=False)
    distinct_docs = Column(Integer, nullable=False)

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

class Document(Base):
    """
    A table row storing the text of a document. Contains the following fields:

    - name: a unique identifier for the document.
    - reader: the process that extracted terms and features from the document.
    - text: a human-readable representation of the document.
    """
    __tablename__ = 'documents'
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, index=True)
    reader = Column(String, nullable=False)
    text = Column(Text, nullable=False)

    def __init__(self, name, reader, text):
        self.name = name
        self.reader = reader
        self.text = text
    
    def __repr__(self):
        return "<Document [%s]: %r>" % (self.reader, self.name)

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
        self.sql_engine = create_engine('sqlite:///'+filename)
        self.sql_session_maker = sessionmaker(bind=self.sql_engine)
        self.sql_session = self.sql_session_maker()
        Base.metadata.create_all(bind=self.sql_engine)

    def _increment_term_count(self, term, value=1, newdoc=False):
        term_entry = self.sql_session.query(Term).get(term)
        if term_entry:
            term_entry.count += value
            if newdoc:
                term_entry.distinct_docs += 1
        else:
            assert newdoc, "Something is wrong -- how did this term "\
                           "appear without appearing in a document?"
            term_entry = Term(term, value, 1, 0)
            self.sql_session.add(term_entry)
        
    def _increment_term_document_count(self, term, document, value=1):
        """
        Increments the count of a term in a document. Returns true if that
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
    
    def increment_term_in_document(self, term, document, value=1):
        newdoc = self._increment_term_document_count(term, document, value)
        absv = math.abs(value)
        self._increment_term_count(term, newdoc, absv)
        newdoc_any = self._increment_term_document_count(ANY, document, absv)
        self._increment_term_count(ANY, newdoc_any, absv)
        self._update_term_relevance(term)
        self.commit()

    def add_document(self, document, terms, text, reader_name):
        """
        Record the terms in a document in the database. If the database already
        has a document with this name, that document will be replaced.

        The terms must already be extracted by some other process.
        `reader_name` indicates which reader was used.
        """
        doc = self.sql_session.query(Document).get(document)
        if doc is not None:
            if doc.text == text and doc.reader == reader_name:
                # nothing has changed, so return
                return
            self.clear_document(document)
        
        for term in terms:
            if isinstance(term, tuple):
                # this is a (term, value) tuple
                term, value = term
            else:
                value = 1
            self.increment_term_in_document(term, document, value)
            

    def clear_document(self, document):
        query = self.sql_session.query(TermInDocument)\
                    .filter(TermInDocument.document == document)
        doc = self.sql_session.query(Document).get(document)
        for row in query.all():
            term, document, count = row
            term_entry = self.sql_session.query(Term).get(term)
            term_entry.count -= count
            term_entry.distinct_docs -= 1
            row.delete()
        doc.delete()
        self.commit()
    
    def count_term(self, term):
        term_entry = self.sql_session.query(Term).get(term)
        if term_entry:
            return term_entry.count
        else:
            return 0

    def term_relevance(self, term):
        words = term.split(' ')
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
    
    def _update_term_relevance(self, term):
        term_entry = self.sql_session.query(Term).get(term)
        term_entry.relevance = self.term_relevance(term)

    def count_term_in_document(self, term, document):
        query = self.sql_session.query(TermInDocument)\
                  .filter(TermInDocument.term == term)\
                  .filter(TermInDocument.document == document)
        try:
            term_entry = query.one()
            return term_entry.count
        except NoResultFound:
            return 0
    
    def count_term_distinct_documents(self, term):
        term_entry = self.sql_session.query(Term).get(term)
        if term_entry:
            return term_entry.distinct_docs
        else:
            return 0

    def count_documents(self):
        return self.count_term_distinct_documents(ANY)
    
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

    def tfidf_term_in_document(self, term, document):
        tf = self.count_term_in_document(term, document)\
           / self.count_term_in_document(ANY, document)
        idf = math.log(1 + self.count_term_distinct_documents(ANY))\
            - math.log(1 + self.count_term_distinct_documents(term))
        return tf * idf

    def commit(self):
        self.sql_session.commit()

