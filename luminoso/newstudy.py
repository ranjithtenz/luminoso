#!/usr/bin/env python
import divisi2
from divisi2.ordered_set import PrioritySet
from luminoso.term_database import TermDatabase
from luminoso.text_readers import get_reader, DOCUMENT
import os
from config import Config

EXTRA_STOPWORDS = [
    'also', 'not', 'without', 'ever', 'because', 'then', 
    'than', 'do', 'just', 'how', 'out', 'much', 'both', 'other'
]

"""
Overall design:
- Luminoso as a whole defines some canonicals that can be easily included
- A LuminosoSpace contains many LuminosoStudies, plus canonicals
- A LuminosoStudy contains many Documents (many of which also go into the space)
- Spaces and studies are configured using the `config` module, giving
  configurations that are both easily human-readable and computer-readable
"""

class LuminosoSpace(object):
    """
    A LuminosoSpace is a meta-study. You supply it with as many documents
    as possible from the space of documents you intend to analyze, or
    possibly other forms of domain-specific knowledge.

    The LuminosoSpace represents the semantic similarities between things
    as a Divisi2 reconstructed association matrix. This matrix can be
    updated incrementally to take new data into account, which is how
    Luminoso learns new domain-specific knowledge.
    """
    CONFIG_FILENAME = 'luminoso.cfg'
    ASSOC_FILENAME = 'associations.rmat'
    DB_FILENAME = 'terms.sqlite'

    def __init__(self, dir):
        """
        A LuminosoSpace is constructed from `dir`, a path to a directory.
        This directory will contain saved versions of various matrices, as
        well as a SQLite database of terms and documents.
        """
        if not os.access(dir, os.R_OK):
            raise IOError("Cannot read the study directory %s. "
                          "Use LuminosoSpace.make() to make a new one.")
        self.dir = dir
        self._load_config()
        self._load_assoc()
        self.database = TermDatabase(
          self.filename_in_dir(LuminosoSpace.DB_FILENAME)
        )
    
    def filename_in_dir(self, filename):
        """
        Given a filename relative to this LuminosoSpace's directory, get its
        complete path.
        """
        return self.dir + os.sep + filename
    
    def file_exists_in_dir(self, filename):
        """
        Determine whether a file exists in this LuminosoSpace's directory.
        """
        return os.access(self.filename_in_dir(filename))

    def _load_config(self):
        "Load the configuration file."
        if self.file_exists_in_dir(LuminosoSpace.CONFIG_FILENAME):
            self.config = Config(
              open(self.filename_in_dir(LuminosoSpace.CONFIG_FILENAME))
            )
        else:
            self.config = self._default_config()
            self.save_config()

    def save_config(self):
        "Save the current configuration to the configuration file."
        out = open(self.filename_in_dir(LuminosoSpace.CONFIG_FILENAME), 'w')
        self.config.save(out)
        out.close()

    def _default_config(self):
        "The default configuration for new studies."
        config = Config()
        config['num_concepts'] = 50000
        config['num_axes'] = 50
        return config

    def _load_assoc(self):
        "Load the association matrix from a file."
        if self.file_exists_in_dir('associations.rmat'):
            self.assoc = divisi2.load('associations.rmat')
        else:
            raise IOError("This LuminosoSpace does not have an "
                          "'associations.rmat' file. Use LuminosoSpace.make() "
                          "to make a valid LuminosoSpace.")
    
    def save_assoc(self):
        "Save the association matrix to a file."
        divisi2.save(self.assoc, self.filename_in_dir('associations.rmat'))

    def train_document(self, docname, text, reader_name, learn=True):
        reader = get_reader(reader_name)
        doc_terms = []
        for weight, term1, term2 in reader.extract_connections(text):
            if term1 == DOCUMENT:
                doc_terms.append((term2, weight))
            else:
                if learn:
                    self.learn_assoc(weight, term1, term2)
        self.database.add_document(docname, doc_terms, text, reader_name)

    def learn_assoc(weight, term1, term2):
        raise NotImplementedError

    @staticmethod
    def make_english():
        """
        Make a LuminosoSpace trained on English common sense.
        """
        assoc = divisi2.network.conceptnet_assoc('en')
        (U, S, _) = assoc.normalize_all().svd(k=100)
        rmat = divisi2.reconstruct_activation(U, S, post_normalize=True)
        return LuminosoSpace(rmat)
