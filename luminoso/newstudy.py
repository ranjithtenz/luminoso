#!/usr/bin/env python
from csc import divisi2
from csc.divisi2.ordered_set import RecyclingSet
from csc.nl import get_nl

EXTRA_STOPWORDS = ['also', 'not', 'without', 'ever', 'because', 'then', 'than', 'do', 'just', 'how', 'out', 'much', 'both', 'other']

# BigramAssocMeasures.likelihood_ratio

class LuminosoSpace(object):
    def __init__(self, rmat, queue=10000):
        """
        A LuminosoSpace is a meta-study. You supply it with as many documents
        as possible from the space of documents you intend to analyze, or
        possibly other forms of domain-specific knowledge.

        The LuminosoSpace represents the semantic similarities between things
        as a Divisi2 reconstructed association matrix. This matrix can be
        updated incrementally to take new data into account, which is how
        Luminoso learns new domain-specific knowledge.

        A LuminosoSpace is constructed from the following parameters:

        - `rmat`: the ReconstructedMatrix to start learning from.
          (This could be the state of a previously-created LuminosoSpace, or
          it could be made from scratch out of common sense, as it is in
          :func:`LuminosoSpace.make_english`.)
        - `queue`: either a RecyclingSet that maintains a queue of recently-
          used terms, or an integer specifying how large the initially empty
          RecyclingSet should be.
        """
        self.rmat = rmat
        if isinstance(queue, int):
            self.queue = RecyclingSet()

    @staticmethod
    def make_english():
        """
        Make a LuminosoSpace trained on English common sense.
        """
        assoc = divisi2.network.conceptnet_assoc('en')
        (U, S, _) = assoc.normalize_all().svd(k=100)
        rmat = divisi2.reconstruct_activation(U, S, post_normalize=True)
        return LuminosoSpace(rmat)
