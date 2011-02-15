#!/usr/bin/env python
from csc import divisi2
from csc.divisi2.ordered_set import RecyclingSet
from csc.nl import get_nl
from luminoso.term_database import TermDatabase
import os

EXTRA_STOPWORDS = ['also', 'not', 'without', 'ever', 'because', 'then', 'than', 'do', 'just', 'how', 'out', 'much', 'both', 'other']

"""
TODO:
- figure out how to load the TermDatabase (which kind of has to be a file)
- 
"""

class LuminosoSpace(object):
    def __init__(self, dir):
        """
        A LuminosoSpace is a meta-study. You supply it with as many documents
        as possible from the space of documents you intend to analyze, or
        possibly other forms of domain-specific knowledge.

        The LuminosoSpace represents the semantic similarities between things
        as a Divisi2 reconstructed association matrix. This matrix can be
        updated incrementally to take new data into account, which is how
        Luminoso learns new domain-specific knowledge.

        A LuminosoSpace is constructed from `dir`, a path to a directory.
        This directory will contain saved versions of various matrices, as
        well as a SQLite database of terms and documents.
        """
        if not os.access(dir, os.R_OK):
            raise IOError("Cannot read the study directory %s. "
                          "Use LuminosoSpace.make() to make a new one.")
        self.dir = dir
        self.rmat = None
        self.termdb = TermDatabase(dir+os.sep+'terms.db')

    @staticmethod
    def make_english():
        """
        Make a LuminosoSpace trained on English common sense.
        """
        assoc = divisi2.network.conceptnet_assoc('en')
        (U, S, _) = assoc.normalize_all().svd(k=100)
        rmat = divisi2.reconstruct_activation(U, S, post_normalize=True)
        return LuminosoSpace(rmat)
