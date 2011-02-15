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
from sqlalchemy import create_engine, Column, Integer, Float, String

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

#class GlobalData(Base):
#    __tablename__ = 'global_data'
#    key = Column(String, primary_key=True)
#    value = Column(Integer)
#
#    def __init__(self, key, value):
#        self.key = key
#        self.value = value

class TermDatabase(object):
    def __init__(self, filename):
        self.sql_engine = create_engine('sqlite:///'+filename)
        self.sql_session_maker = sessionmaker(bind=self.sql_engine)
        self.sql_session = self.sql_session_maker()
        Base.metadata.create_all(bind=self.sql_engine)

    def _increment_term_count(self, term, newdoc=False):
        term_entry = self.sql_session.query(Term).get(term)
        if term_entry:
            term_entry.count += 1
            if newdoc:
                term_entry.distinct_docs += 1
        else:
            assert newdoc, "Something is wrong -- how did this term "\
                           "appear without appearing in a document?"
            term_entry = Term(term, 1, 1, 0)
            self.sql_session.add(term_entry)
        
    def _increment_term_document_count(self, term, document):
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
            term_entry.count += 1
            return False
        except NoResultFound:
            term_entry = TermInDocument(term, document, 1)
            self.sql_session.add(term_entry)
            return True
    
    def increment_term_in_document(self, term, document):
        newdoc = self._increment_term_document_count(term, document)
        self._increment_term_count(term, newdoc)
        newdoc_any = self._increment_term_document_count(ANY, document)
        self._increment_term_count(ANY, newdoc_any)
        self._update_term_relevance(term)
    
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

