#!/usr/bin/env python
"""
This module defines the LuminosoSpace object in luminoso2.

Overall design:
- Luminoso as a whole defines some canonicals that can be easily included
- A LuminosoSpace contains many LuminosoStudies, plus canonicals
- A LuminosoStudy contains many Documents (many of which also go into the space)
- Spaces and studies are configured using the `config` module, giving
  configurations that are both easily human-readable and computer-readable
"""

from __future__ import with_statement   # for Python 2.5 support
import divisi2
from divisi2.reconstructed import ReconstructedMatrix
from divisi2.ordered_set import PrioritySet
from luminoso.term_database import TermDatabase, _BIG
from luminoso.text_readers import get_reader, DOCUMENT
import os
from config import Config

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

    def __init__(self, space_dir):
        """
        A LuminosoSpace is constructed from `dir`, a path to a directory.
        This directory will contain saved versions of various matrices, as
        well as a SQLite database of terms and documents.
        """
        if not os.access(space_dir, os.R_OK):
            raise IOError("Cannot read the study directory %s. "
                          "Use LuminosoSpace.make() to make a new one."
                          % space_dir)
        self.dir = space_dir
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
        return os.access(self.filename_in_dir(filename), os.F_OK)

    def _load_config(self):
        "Load the configuration file."
        if self.file_exists_in_dir(LuminosoSpace.CONFIG_FILENAME):
            self.config = Config(
              open(self.filename_in_dir(LuminosoSpace.CONFIG_FILENAME))
            )
        else:
            self.config = _default_config()
            self.save_config()

    def save_config(self):
        "Save the current configuration to the configuration file."
        out = open(self.filename_in_dir(LuminosoSpace.CONFIG_FILENAME), 'w')
        self.config.save(out)
        out.close()

    def _load_assoc(self):
        "Load the association matrix and priority queue from a file."
        if self.file_exists_in_dir('associations.rmat'):
            self.assoc = divisi2.load(self.filename_in_dir('associations.rmat'))
            assert isinstance(self.assoc, ReconstructedMatrix)
        else:
            raise IOError("This LuminosoSpace does not have an "
                          "'associations.rmat' file. Use LuminosoSpace.make() "
                          "to make a valid LuminosoSpace.")
        if not isinstance(self.assoc.row_labels, PrioritySet):
            # turn an ordinary ReconstructedMatrix into one that has
            # PrioritySets for indices
            priority = PrioritySet(self.config['num_concepts'])
            items = self.assoc.row_labels
            item_tuples = zip(items, [_BIG] * len(items))
            priority.load_items(item_tuples)
            self.assoc.row_labels = priority
            self.assoc.col_labels = priority

        self.priority = self.assoc.row_labels
        self.priority.listen_for_drops(self.on_drop)

    def save_assoc(self):
        "Save the association matrix to a file."
        divisi2.save(self.assoc, self.filename_in_dir('associations.rmat'))
    
    def on_drop(self, index, key):
        """
        Handle when a key falls out of the PrioritySet.
        """
        self.assoc.left[index, :] = 0

    def add_document(self, doc, reader_name=None):
        """
        Take in a document, pass it through the reader, and store its terms
        in the term database.

        The document should be expressed as a dictionary, containing at least
        these keys:
        - name: the unique identifier for the document
        - text: the plain text of the document, possibly including text-encoded
          tags

        Optionally, it may contain:
        - tags: (key, value) tuples representing tags
        """
        if reader_name is None:
            reader_name = self.config['reader']
        reader = get_reader(reader_name)
        docname = doc['name']
        text = doc['text']
        doc_terms = []
        for weight, term1, term2 in reader.extract_connections(text):
            if term1 == DOCUMENT:
                doc_terms.append((term2, weight))
                relevance = self.database.term_relevance(term2)
                self.priority.add(term2)
                self.priority.update(term2, relevance)
        self.database.add_document(docname, doc_terms, text, reader_name)
        for key, value in doc.get('tags', []):
            self.database.set_tag_on_document(docname, key, value)

    def learn_document(self, docname):
        """
        Given a previously added document, use it to update the association
        matrix. This can be repeated to increase accuracy.
        """
        doc = self.database.get_document(docname)
        reader = get_reader(doc.reader)
        for weight, term1, term2 in reader.extract_connections(doc.text):
            if term1 != DOCUMENT:
                self.learn_assoc(weight, term1, term2)

    def learn_assoc(self, weight, term1, term2):
        """
        Learn the strength of the association between term1 and term2.
        """
        row = self.assoc.row_labels.add(term1)
        col = self.assoc.col_labels.add(term2)
        self.assoc[row, col] = weight          # do a Hebbian step

    def __repr__(self):
        return "<LuminosoSpace: %r>" % self.dir

    @staticmethod
    def make(space_dir, rmat):
        """
        Make a new LuminosoSpace in the (nonexistent) directory `dir`,
        with initial association matrix `rmat`.
        """
        os.mkdir(space_dir)
        rmat_file = space_dir + os.sep + 'associations.rmat'
        divisi2.save(rmat_file, rmat)
        return LuminosoSpace(space_dir)

    @staticmethod
    def make_english(space_dir):
        """
        Make a LuminosoSpace whose initial matrix contains English common sense.
        """
        assoc = divisi2.network.conceptnet_assoc('en')
        (mat_U, diag_S, _) = assoc.normalize_all().svd(k=100)
        rmat = divisi2.reconstruct_activation(
            mat_U, diag_S, post_normalize=True
        )
        return LuminosoSpace.make(space_dir, rmat)

def _default_config():
    "The default configuration for new studies."
    config = Config()
    # FIXME: so far we just assume that these will match the initial rmat
    config['num_concepts'] = 50000
    config['num_axes'] = 100
    config['reader'] = 'simplenlp.en'
    return config

