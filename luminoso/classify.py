"""
Classify terms and documents (in general, vectors) as being associated with a canonical document or not.

Since terms or documents can be associated with several or no canonical documents, a one-class SVM is used.
"""
import numpy as np
from scikits.learn import svm

class Classifier(object):
    def __init__(self, vectors, **params):
        params.setdefault('kernel', 'linear')
        self.clf = svm.OneClassSVM(**params)

        # Scale vectors
        training = np.asarray(vectors)
        self.mean = np.mean(training, axis=0)
        training -= self.mean
        self.std = training.std(axis=0)
        training /= self.std
        self.clf.fit(training)

    def _scale(self, vectors):
        return (np.asarray(vectors) - self.mean) / self.std

    def distances(self, vectors):
        return self.clf.decision_function(self._scale(vectors))

    def closest_terms(self, model, terms, n):
        vecs = vectors_for_terms(model, terms)
        distances = self.distances(vecs)
        closest_idxes = np.argsort(np.abs(distances.ravel()))
        return [(terms[i], distances[i]) for i in closest_idxes[:n]]

    def predict(self, vectors):
        return self.clf.predict(self._scale(vectors))


def vectors_for_terms(model, terms):
    return [model.vector_from_terms([(term, 1.0)]) for term in terms]

