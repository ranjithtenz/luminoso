#!/usr/bin/env python
"""
This module defines the LuminosoModel object in luminoso2.

Overall design:
- Luminoso as a whole defines some canonicals that can be easily included
- A LuminosoModel contains many LuminosoStudies, plus canonicals
- A LuminosoStudy contains many Documents (many of which also go into the model)
- Spaces and studies are configured using the `config` module, giving
  configurations that are both easily human-readable and computer-readable
"""

from __future__ import with_statement   # for Python 2.5 support
import divisi2
from divisi2.fileIO import load_pickle, save_pickle
from divisi2.reconstructed import ReconstructedMatrix
from divisi2.ordered_set import PrioritySet
from luminoso.term_database import TermDatabase, _BIG
from luminoso.text_readers import get_reader, DOCUMENT, TAG
import os
from config import Config
import logging
LOG = logging.getLogger(__name__)

class LuminosoModel(object):
    """
    A LuminosoModel is a meta-study. You supply it with as many documents
    as possible from the space of documents you intend to analyze, or
    possibly other forms of domain-specific knowledge.

    The LuminosoModel represents the semantic similarities between things
    as a Divisi2 reconstructed association matrix. This matrix can be
    updated incrementally to take new data into account, which is how
    Luminoso learns new domain-specific knowledge.
    """
    CONFIG_FILENAME = 'luminoso.cfg'
    ASSOC_FILENAME = 'associations.rmat'
    DB_FILENAME = 'terms.sqlite'

    def __init__(self, model_dir):
        """
        A LuminosoModel is constructed from `dir`, a path to a directory.
        This directory will contain saved versions of various matrices, as
        well as a SQLite database of terms and documents.
        """
        if not os.access(model_dir, os.R_OK):
            raise IOError("Cannot read the study directory %s. "
                          "Use LuminosoModel.make() to make a new one."
                          % model_dir)
        self.dir = model_dir
        self._load_config()
        self._load_assoc()
        self.database = TermDatabase(
          self.filename_in_dir(LuminosoModel.DB_FILENAME)
        )
    
    def filename_in_dir(self, filename):
        """
        Given a filename relative to this LuminosoModel's directory, get its
        complete path.
        """
        return self.dir + os.sep + filename
    
    def file_exists_in_dir(self, filename):
        """
        Determine whether a file exists in this LuminosoModel's directory.
        """
        return os.access(self.filename_in_dir(filename), os.F_OK)

    def _load_config(self):
        "Load the configuration file."
        if self.file_exists_in_dir(LuminosoModel.CONFIG_FILENAME):
            self.config = Config(
              open(self.filename_in_dir(LuminosoModel.CONFIG_FILENAME))
            )
        else:
            self.config = _default_config()
            self.save_config()

    def save_config(self):
        "Save the current configuration to the configuration file."
        out = open(self.filename_in_dir(LuminosoModel.CONFIG_FILENAME), 'w')
        self.config.save(out)
        out.close()

    def _load_assoc(self):
        "Load the association matrix and priority queue from a file."
        if self.file_exists_in_dir('associations.rmat'):
            self.assoc = load_pickle(
                self.filename_in_dir('associations.rmat')
            )
            assert isinstance(self.assoc, ReconstructedMatrix)
        else:
            raise IOError("This LuminosoModel does not have an "
                          "'associations.rmat' file. Use LuminosoModel.make() "
                          "to make a valid LuminosoModel.")
        self.assoc.make_symmetric()
        if not isinstance(self.assoc.row_labels, PrioritySet):
            # turn an ordinary ReconstructedMatrix into one that has
            # PrioritySets for indices
            priority = PrioritySet(self.config['num_concepts'])
            if self.assoc.row_labels is not None:
                items = self.assoc.row_labels
                item_tuples = zip(items, [_BIG] * len(items))
                priority.load_items(item_tuples)
            self.assoc.set_symmetric_labels(priority)

        self.priority = self.assoc.row_labels
        self.priority.listen_for_drops(self.on_drop)

    def save_assoc(self):
        "Save the association matrix to a file."
        save_pickle(self.assoc, self.filename_in_dir('associations.rmat'))
    
    def on_drop(self, index, key):
        """
        Handle when a key falls out of the PrioritySet.
        """
        self.assoc.left[index, :] = 0
        self.database.clear_term_priority_index(key)

    def add_document(self, doc, reader_name=None):
        """
        Take in a document, pass it through the reader, and store its terms
        in the term database.

        The document should be expressed as a dictionary, containing at least
        these keys:
        - name: the unique identifier for the document
        - text: the plain text of the document, possibly including text-encoded
          tags
        - url: a unique identifier for the document, preferably one that
          actually locates it relative to the study

        Optionally, it may contain:
        - tags: (key, value) tuples representing tags
        """
        LOG.info("Reading document: %r" % doc['url'])
        if reader_name is None:
            reader_name = self.config['reader']
        reader = get_reader(reader_name)
        text = doc['text']
        tags = doc.get('tags', [])
        doc_terms = []
        for weight, term1, term2 in reader.extract_connections(text):
            if term1 == DOCUMENT:
                if isinstance(term2, tuple) and term2[0] == TAG:
                    tags.append(term2[1:])
                else:
                    doc_terms.append((term2, weight))
                    relevance = self.database.term_relevance(term2)
                    self.priority.add(term2)
                    self.priority.update(term2, relevance)

        doc['reader'] = reader_name
        doc['terms'] = doc_terms
        doc['tags'] = tags
        self.database.add_document(doc)
        self.database.find_term_texts(text, reader)

    def learn_document(self, docid):
        """
        Given a previously added document, use it to update the association
        matrix. This can be repeated to increase accuracy.
        """
        doc = self.database.get_document(docid)
        reader = get_reader(doc.reader)
        for weight, term1, term2 in reader.extract_connections(doc.text):
            if term1 != DOCUMENT:
                self.learn_assoc(weight, term1, term2)

    def learn_assoc(self, weight, term1, term2):
        """
        Learn the strength of the association between term1 and term2.
        """
        row = self.assoc.row_labels.add(term1)
        self.database.set_term_priority_index(term1, row)
        col = self.assoc.col_labels.add(term2)
        self.database.set_term_priority_index(term2, col)
        mse = self.assoc.hebbian_step(row, col, weight)
        return mse

    def __repr__(self):
        return "<LuminosoModel: %r>" % self.dir

    @staticmethod
    def make(model_dir, orig_dmat, config):
        """
        Make a new LuminosoModel in the (nonexistent) directory `dir`,
        with initial half-association matrix `orig_dmat`. (A half-association
        matrix is a matrix that gives an association matrix when it is
        multiplied by its transpose.)
        """
        rows = config['num_concepts']
        cols = config['num_axes']
        if orig_dmat.shape != (rows, cols):
            dmat = divisi2.DenseMatrix((rows, cols))
            rows_to_copy = orig_dmat.left.shape[0]
            if rows < rows_to_copy:
                raise ValueError("num_concepts is too small to fit the "
                                 "existing concepts.")
            cols_to_copy = min(cols, orig_dmat.left.shape[1])
            dmat[:rows_to_copy, :cols_to_copy] = \
              orig_dmat[:rows_to_copy, :cols_to_copy]
            dmat.row_labels = orig_dmat.row_labels
        else:
            dmat = orig_dmat
        
        rmat = divisi2.reconstruct_symmetric(dmat)
        os.mkdir(model_dir)
        rmat_file = model_dir + os.sep + 'associations.rmat'
        save_pickle(rmat, rmat_file)
        return LuminosoModel(model_dir)

    @staticmethod
    def make_empty(model_dir, config=None):
        """
        Make a LuminosoModel that starts from an empty matrix.
        """
        if config is None:
            config = _default_config()
        mat = divisi2.DenseMatrix((config['num_concepts'], config['num_axes']))
        model = LuminosoModel.make(model_dir, mat, config)
        return model

    @staticmethod
    def make_english(model_dir, config=None):
        """
        Make a LuminosoModel whose initial matrix contains English common sense.
        """
        if config is None:
            config = _default_config()
        if os.access(model_dir, os.F_OK):
            raise IOError("The model directory %r already exists." % model_dir)
        assoc = divisi2.network.conceptnet_assoc('en')
        (mat_U, diag_S, _) = assoc.normalize_all().svd(k=100)
        rmat = divisi2.reconstruct_activation(
            mat_U, diag_S, post_normalize=True
        )
        model = LuminosoModel.make(model_dir, rmat.left, config)
        model.config['iteration'] = 1000
        return model

def _default_config():
    "The default configuration for new studies."
    config = Config()
    # FIXME: so far we just assume that these will match the initial rmat
    config['num_concepts'] = 50000
    config['num_axes'] = 100
    config['reader'] = 'simplenlp.en'
    config['iteration'] = 0
    return config

