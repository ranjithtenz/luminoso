"""
This file interacts with a SQLite file that keeps a running count of the
number of times various terms appear, and the number of times they appear
in each document.
"""
#from nltk.metrics import BigramAssocMeasures
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy import create_engine, Column, Integer, String

Base = declarative_base()
class Term(Base):
    __tablename__ = 'terms'
    term = Column(String, primary_key=True)
    count = Column(Integer, nullable=False)

    def __init__(self, term, count):
        self.term = term
        self.count = count
    
    def __repr__(self):
        return "<%r, %d occurrences>" % (self.term, self.count)

class TermInDocument(Base):
    __tablename__ = 'document_terms'
    id = Column(Integer, primary_key=True)
    term = Column(String, nullable=False)
    document = Column(String, nullable=False)
    count = Column(Integer, nullable=False)

    def __init__(self, term, document, count):
        self.term = term
        self.document = document
        self.count = count

    def __repr__(self):
        return "<%r in %r, %d occurrences>" % (self.term,
                                               self.document, 
                                               self.count)

class TermCounter(object):
    def __init__(self, filename):
        self.sql_engine = create_engine('sqlite:///'+filename)
        self.sql_session = sessionmaker(bind=self.sql_engine)

    def _increment_term_count(self, term):
        query = self.sql_session.query(Term).filter(Term.term == term)
        try:
            term_entry = query.one()
            term_entry.count += 1
        except NoResultFound:
            term_entry = Term(term, 1)
            self.sql_session.add(term_entry)

    def _increment_term_document_count(self, term, document):
        query = self.sql_session.query(TermInDocument)\
                  .filter(TermInDocument.term == term,
                          TermInDocument.document == document)
        try:
            term_entry = query.one()
            term_entry.count += 1
        except NoResultFound:
            term_entry = TermInDocument(term, document, 1)
            self.sql_session.add(term_entry)

