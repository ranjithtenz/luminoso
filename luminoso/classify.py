"""
Classify terms and documents (in general, vectors) as being associated with a canonical document or not.

Since terms or documents can be associated with several or no canonical documents, a one-class SVM is used.
"""
import numpy as np
from scikits.learn import svm

class BaseClassifier(object):
    def _scaled_training_data(self, vectors):
        training = np.asarray(vectors)
        self.mean = np.mean(training, axis=0)
        training -= self.mean
        self.std = training.std(axis=0)
        training /= self.std
        return training

    def _scale(self, vectors):
        return (np.asarray(vectors) - self.mean) / self.std


class OneClassClassifier(BaseClassifier):
    def __init__(self, vectors, **params):
        params.setdefault('kernel', 'linear')
        self.clf = svm.OneClassSVM(**params)
        self.clf.fit(self._scaled_training_data(training))

    def distances(self, vectors):
        return self.clf.decision_function(self._scale(vectors))

    def closest_terms(self, model, terms, n):
        vecs = vectors_for_terms(model, terms)
        distances = self.distances(vecs)
        closest_idxes = np.argsort(np.abs(distances.ravel()))
        return [(terms[i], distances[i]) for i in closest_idxes[:n]]

    def predict(self, vectors):
        return self.clf.predict(self._scale(vectors))

class NClassifier(BaseClassifier):
    def __init__(self, vectors, classes, **params):
        params.setdefault('kernel', 'linear')
        self.clf = svm.SVC(probability=True)
        self.clf.fit(self._scaled_training_data(vectors), classes)

    def top_classes(self, vector, n=5):
        probs = self.clf.predict_proba([vector]).ravel()
        order = np.argsort(probs)
        return [(i, probs[i]) for i in reversed(order[-5:])]


def vectors_for_terms(model, terms):
    return [model.vector_from_terms([(term, 1.0)]) for term in terms]

