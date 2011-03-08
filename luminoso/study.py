#!/usr/bin/env python
from __future__ import with_statement
"""
This class provides the model to SVDView's view, calculating a blend of all
the components of a study that it finds in the filesystem.
"""
import sys, os
if __name__ == '__main__':
    # prepare for other imports
    sys.path.extend([os.path.join(os.path.dirname(sys.argv[0]), "lib"),
                     os.path.dirname(sys.argv[0])])

import os, codecs, time
import cPickle as pickle
import numpy as np
import traceback
import logging
import hashlib
import chardet
logger = logging.getLogger('luminoso')

from standalone_nlp.lang_en import en_nl
from csc import divisi2
from csc.divisi2.blending import blend
from csc.divisi2.ordered_set import OrderedSet

from luminoso.whereami import package_dir
from luminoso.report import render_info_page, default_info_page

import shutil

# Warning: SUBTRACT_MEAN might screw up statistics. We don't really know 
# what is going on.
SUBTRACT_MEAN = False

EXTRA_STOPWORDS = ['also', 'not', 'without', 'ever', 'because', 'then', 'than', 'do', 'just', 'how', 'out', 'much', 'both', 'other']

try:
    import json
except ImportError:
    import simplejson as json

class OutdatedAnalysisError(Exception):
    pass

class Document(object):
    '''
    A Document is an entity in a Study.
    '''
    def __init__(self, name, text):
        self.name = name
        self.text = text

    @classmethod
    def from_file(cls, filename, name):
        # Open in text mode.
        rawtext = open(filename, 'r')
        encoding = chardet.detect(rawtext.read())['encoding']
        rawtext.close()
        text = codecs.open(filename, encoding=encoding, errors='replace').read()
        return cls(name, text)

    def extract_concepts_with_negation(self):
        return extract_concepts_with_negation(self.text)

    def get_sentences(self):
        words = en_nl.tokenize(self.text).split()
        sentences = []
        current_sentence = []
        for word in words:
            if word in PUNCTUATION:
                if len(current_sentence) >= 1:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                current_sentence.append(word)
        sentences.append(current_sentence)
        return sentences

class CanonicalDocument(Document):
    pass


NEGATION = ['no', 'not', 'never', 'stop', 'lack', "n't", "without"]
PUNCTUATION = ['.', ',', '!', '?', '...', '-', ':', ';', '``', "''", "`", "'"]
def extract_concepts_with_negation(text):
    words = en_nl.tokenize(text).split()
    return extract_concepts_from_words(words)

def extract_concepts_from_words(words):
    # FIXME: this may join together words from different contexts...
    positive_words = []
    negative_words = []
    neg_tagged_words = []
    pos_tagged_words = []
    positive = True
    for word in words:
        if word.startswith('#-'):
            neg_tagged_words.append('#'+word[2:])
        elif word.startswith('#'):
            pos_tagged_words.append(word)
        elif word.lower() in NEGATION:
            positive = False
        elif word.lower() in EXTRA_STOPWORDS: continue
        else:
            if positive:
                positive_words.append(word)
            else:
                negative_words.append(word)
            if word in PUNCTUATION:
                positive = True
    positive_concepts = [(c, 1) for c in en_nl.extract_concepts(' '.join(positive_words))]
    negative_concepts = [(c, -1) for c in en_nl.extract_concepts(' '.join(negative_words))]
    neg_tagged_concepts = [(c, -1) for c in neg_tagged_words]
    pos_tagged_concepts = [(c, 1) for c in pos_tagged_words]
    return positive_concepts + pos_tagged_concepts + negative_concepts + neg_tagged_concepts

def load_json_from_file(file):
    with open(file) as f:
        return json.load(f)

def write_json_to_file(data, file):
    with open(file, 'w') as f:
        json.dump(data, f)
        
def entry_count(vec):
    return np.sum(np.abs(vec))

DEFAULT_SETTINGS = {
    'axes': 50,
    'concept_cutoff': 2
}

class Study(object):
    '''
    A Study is a collection of documents and other matrices that can be analyzed.
    '''
    def __init__(self, name, documents, canonical, other_matrices, settings, status_callback=None):
        """
        documents: list of Document objects
        canonical: list of Document objects that are the canonical documents (possibly empty)
        other_matrices: things to blend.
        settings: a dict of settings. See DEFAULT_SETTINGS above.
        status_callback: called with a status message to report progress. (Can be None.)
        """
        self.name = name
        self.study_documents = documents
        self.canonical_documents = canonical
        # self.documents is now a property
        self._documents_matrix = None
        self.other_matrices = other_matrices
        self.settings = settings
        self.status_callback = status_callback

    def config(self, key):
        if key in self.settings: return self.settings[key]
        else: return DEFAULT_SETTINGS[key]
        
    def _step(self, msg):
        logger.info(msg)
        if self.status_callback is not None:
            self.status_callback(msg)

    def get_contents_hash(self):
        def sha1(txt):
            if isinstance(txt, unicode): txt = txt.encode('utf-8')
            return hashlib.sha1(txt).hexdigest()

        docs = dict((doc.name, (isinstance(doc, CanonicalDocument),
                                sha1(doc.text)))
                    for doc in self.documents)
        matrices = tuple(sorted(self.other_matrices.keys()))

        # TODO: make sure matrices have a meaningful `hash`.
        #matrices = dict((name, hash(mat)) for name, mat in self.other_matrices.items())
        return dict(docs=docs, matrices=matrices)

    @property
    def num_documents(self):
        return len(self.documents)

    @property
    def documents(self):
        return self.study_documents + self.canonical_documents
    
    def get_documents_matrix(self):
        """
        Get a matrix of documents vs. concepts.

        This is temporarily cached (besides what StudyDir does) because it
        will be needed multiple times in analyzing a study.

        FIXME: try to make canonical documents not change the results
        """
        self._step('Building document matrix...')
        if self.num_documents == 0:
            assert False
            return None
        if self._documents_matrix is not None:
            return self._documents_matrix
        entries = []
        for doc in self.study_documents:
            self._step(doc.name)
            for concept, value in doc.extract_concepts_with_negation()[:1000]:
                if (concept not in PUNCTUATION) and (not en_nl.is_blacklisted(concept)):
                    entries.append((value, doc.name, concept))
        documents_matrix = divisi2.make_sparse(entries).normalize_tfidf(cols_are_terms=True)
        canon_entries = []
        for doc in self.canonical_documents:
            self._step(doc.name)
            for concept, value in doc.extract_concepts_with_negation()[:1000]:
                if (concept not in PUNCTUATION) and (not en_nl.is_blacklisted(concept)):
                    canon_entries.append((value, doc.name, concept))
        if canon_entries:
            canonical_matrix = divisi2.make_sparse(canon_entries).normalize_rows()
            self._documents_matrix = documents_matrix + canonical_matrix
        else:
            self._documents_matrix = documents_matrix
        return self._documents_matrix
    
    def get_documents_assoc(self):
        self._step('Finding associated concepts...')
        if self.num_documents == 0: return None
        docs = self.get_documents_matrix()
        concept_counts = docs.col_op(len)
        valid_concepts = set()

        # NOTE: this is the number you change to make a study larger or
        # smaller.
        for concept, count in concept_counts.to_sparse().named_items():
            if count >= self.config('concept_cutoff'): valid_concepts.add(concept)
        if len(valid_concepts) == 0:
            # No valid concepts. This unfortunately happens when
            # concept_cutoff is too low.
            return None

        entries = []
        for doc in self.study_documents:
            prev_concepts = []
            for sentence in doc.get_sentences():
                # avoid insane space usage by limiting to 20 words
                concepts = extract_concepts_from_words(sentence[:20])
                for concept1, value1 in concepts:
                    if concept1 in valid_concepts:
                        for concept2, value2 in concepts:
                            if concept2 in valid_concepts and concept1 < concept2:
                                entries.append( (value1*value2, concept1, concept2) )
                                entries.append( (value1*value2, concept2, concept1) )
                        for concept2, value2 in prev_concepts:
                            if concept2 in valid_concepts and concept1 != concept2:
                                entries.append( (value1*value2/2, concept1, concept2) )
                                entries.append( (value1*value2/2, concept2, concept1) )
                # Remember tags, but forget words that were too long ago
                prev_concepts = [p for p in prev_concepts[:-100] if
                p[0].startswith('#')] + prev_concepts[-100:]
                prev_concepts.extend(concepts)
        assert len(entries) > 0
        return divisi2.SparseMatrix.square_from_named_entries(entries).squish()
    
    def get_blend(self):
        if self.is_associative():
            return self.get_assoc_blend()
        else:
            return self.get_analogy_blend()
    
    def is_associative(self):
        if not self.other_matrices: return True
        return any(name.endswith('.assoc.smat') for name in
                   self.other_matrices)

    def get_analogy_blend(self):
        other_matrices = [matrix for name, matrix in
        self.other_matrices.items() if name.endswith('.smat')]
        other_matrices = self.other_matrices.values()
        
        # find concepts used at least twice
        docs = self.get_documents_matrix()
        concept_counts = docs.col_op(len)
        valid_concepts = set()
        for concept, count in concept_counts.to_sparse().named_items():
            if count >= 3: valid_concepts.add(concept)
        
        # extract relevant concepts from the doc matrix;
        # transpose it so it's concepts vs. documents
        orig_doc_matrix = self.get_documents_matrix()
        #sdoc_indices = [orig_doc_matrix.row_index(sdoc.name)
        #                for sdoc in self.study_documents]
        concept_indices = [orig_doc_matrix.col_index(c)
                           for c in valid_concepts]

        # NOTE: canonical documents can affect the stats this way.
        # Is there a clean way to fix this?

        doc_matrix = orig_doc_matrix[:,concept_indices].T.squish()
        if doc_matrix is None:
            theblend = blend(other_matrices)
            study_concepts = set(theblend.row_labels)
        else:
            theblend = blend([doc_matrix] + other_matrices)
            study_concepts = set(doc_matrix.row_labels)
        return theblend, study_concepts

    def get_assoc_blend(self):
        other_matrices = []
        doc_matrix = self.get_documents_assoc()
        self._step('Blending...')
        for name, matrix in self.other_matrices.items():
            # use association matrices only
            # (unless we figure out how to do both kinds of blending)
            if name.endswith('.assoc.smat'):
                if matrix.shape[0] != matrix.shape[1]:
                    raise ValueError("The matrix %s is not square" % name)
                other_matrices.append(matrix)

        if doc_matrix is None:
            theblend = blend(other_matrices)
            study_concepts = set(theblend.row_labels)
        else:
            theblend = blend([doc_matrix] + other_matrices)
            study_concepts = set(doc_matrix.row_labels)
        return theblend, study_concepts

    def get_eigenstuff(self):
        self._step('Finding eigenvectors...')
        document_matrix = self.get_documents_matrix()
        theblend, study_concepts = self.get_blend()
        U, Sigma, V = theblend.normalize_all().svd(k=self.config('axes'))
        indices = [U.row_index(concept) for concept in study_concepts]
        reduced_U = U[indices]
        if self.is_associative():
            doc_rows = divisi2.aligned_matrix_multiply(document_matrix, reduced_U)
            projections = reduced_U.extend(doc_rows)

        else:
            doc_indices = [V.row_index(doc.name)
                           for doc in self.documents
                           if doc.name in V.row_labels]
            projections = reduced_U.extend(V[doc_indices])
        
        #if SUBTRACT_MEAN:
        #    sdoc_indices = [projections.row_index(doc.name) for doc in
        #    self.study_documents if doc.name in projections.row_labels]
        #    projections -= np.asarray(projections[sdoc_indices]).mean(axis=0)
        if SUBTRACT_MEAN:
            projections -= np.asarray(projections).mean(axis=0)

        return document_matrix, projections, Sigma

    def compute_stats(self, docs, spectral):
        """
        Calculate statistics.

        Consistency: how tightly-clustered the documents are in the spectral
        decomposition space.

        Centrality: a Z-score for how "central" each concept and document
        is. Same general idea as "congruence" from Luminoso 1.0.
        """

        if len(self.study_documents) <= 1:
            # consistency and centrality are undefined
            consistency = None
            centrality = None
            correlation = None
            core = None
            key_concepts = None
            c_centrality = None
            c_correlation = None
        else:
            # Determine which indices of the association matrix correspond to
            # documents.
            doc_indices = [spectral.row_index(doc.name)
                           for doc in self.study_documents
                           if doc.name in spectral.row_labels]
            valid_concepts = [c for c in spectral.row_labels if not c.endswith('.txt')]
            concept_indices = [spectral.row_index(c) for c in valid_concepts]
            
            # Make an ad hoc category of documents, then find how much each
            # document is associated with this average document.
            category_vec = divisi2.DenseVector(spectral.shape[0], spectral.row_labels)
            category_vec[doc_indices] = 1.0/len(doc_indices)
            all_assoc = spectral.left_category(category_vec)
            doc_assoc = all_assoc[doc_indices]
            
            # Calculate similarity statistics over all documents.
            doc_mean = np.mean(np.asarray(doc_assoc))
            doc_stdev = np.std(np.asarray(doc_assoc))
            doc_stderr = doc_stdev / np.sqrt(len(doc_indices))

            # ...and over all concepts, though we may not need this.
            all_mean = np.mean(np.asarray(all_assoc))
            all_stdev = np.std(np.asarray(all_assoc))
            all_stderr = all_stdev / np.sqrt(spectral.shape[0],)

            consistency = doc_mean / doc_stderr
            centrality = divisi2.DenseVector((all_assoc - doc_mean) / doc_stderr, spectral.row_labels)
            correlation = divisi2.DenseVector(all_assoc / doc_stderr, spectral.row_labels)
            core = centrality.top_items(len(centrality)/2)
            core = [c[0] for c in core
                    if c[0] in valid_concepts
                    and c[1] > .001][:20]

            c_centrality = {}
            c_correlation = {}
            key_concepts = {}
            
            # the number of times each concept appears in each document
            doc_occur = self._documents_matrix

            # the average number of occurrences you expect of each document
            baseline = (1.0 + doc_occur.col_op(len)) / doc_occur.shape[0]
            for doc in self.canonical_documents:
                # record centrality and correlation for this document
                c_centrality[doc.name] = centrality.entry_named(doc.name)
                c_correlation[doc.name] = correlation.entry_named(doc.name)

                # find a weighted vector of similar documents
                docvec = np.maximum(0, spectral.row_named(doc.name)[doc_indices]) ** 3
                docvec /= (0.0001 + np.sum(docvec))
                keyvec = divisi2.aligned_matrix_multiply(docvec, doc_occur)

                assert not any(np.isnan(keyvec))
                assert not any(np.isinf(keyvec))
                interesting = spectral.row_named(doc.name)[concept_indices]
                #interesting = keyvec/baseline
                key_concepts[doc.name] = []
                for key, val in interesting.top_items(5):
                    if val > 0.0 and keyvec.entry_named(key) > 0.0:
                        key_concepts[doc.name].append((key, keyvec.entry_named(key)))
        
        return {
            'num_documents': self.num_documents,
            'num_concepts': spectral.shape[0] - self.num_documents,
            'consistency': consistency,
            'centrality': c_centrality,
            'correlation': c_correlation,
            'key_concepts': key_concepts,
            'core': core,
            'timestamp': list(time.localtime())
        }
    
    def analyze(self):
        # TODO: make it possible to blend multiple directories
        self._documents_matrix = None
        docs, projections, Sigma = self.get_eigenstuff()
        magnitudes = np.sqrt(np.sum(np.asarray(projections*projections), axis=1))
        if self.is_associative():
            spectral = divisi2.reconstruct_activation(projections, Sigma, post_normalize=True, offset=0.0001)
        else:
            spectral = divisi2.reconstruct_similarity(projections, Sigma,
            post_normalize=True, offset=0.0001)
        self._step('Calculating stats...')
        stats = self.compute_stats(docs, spectral)
        
        results = StudyResults(self, docs, spectral.left, spectral, magnitudes, stats)
        return results

class StudyResults(object):
    def __init__(self, study, docs, projections, spectral, magnitudes, stats):
        self.study = study
        self.docs = docs
        self.spectral = spectral
        self.projections = projections
        self.magnitudes = magnitudes
        self.stats = stats
        self.canonical_filenames = [doc.name for doc in study.canonical_documents]
        self.info = render_info_page(self)

    def write_coords_as_csv(self, filename):
        # FIXME: not divisi2 ready
        raise NotImplementedError

        import csv
        x_axis = self.projections['DefaultXAxis',:].hat()
        y_axis = self.projections['DefaultYAxis',:].hat()
        output = open(filename, 'w')
        writer = csv.writer(output)
        writer.writerow(['Concept', 'X projection', 'Y projection', 'Coordinates'])
        for concept in self.study_concepts:
            xproj = self.projections[concept,:] * x_axis
            yproj = self.projections[concept,:] * y_axis
            coords = self.projections[concept,:].values()
            row = [concept.encode('utf-8'), xproj, yproj] + coords
            writer.writerow(row)
        output.close()

    def write_report(self, filename):
        if self.stats is None: return
        self.info = render_info_page(self)
        with open(filename, 'w') as out:
            out.write(self.info.encode('utf-8'))
        return self.info

    def write_core(self, filename):
        if self.stats is None: return
        core_str = u', '.join(self.stats['core'])
        with open(filename, 'w') as out:
            out.write(core_str.encode('utf-8')+'\n')

    def get_consistency(self):
        return self.stats['consistency']

    def get_congruence(self, concept):
        return self.stats['congruence'][concept]

    def get_info(self):
        if self.info is not None: return self.info
        else: return default_info_page(self.study)

    def get_concept_info(self, concept):
        if concept not in self.spectral.row_labels: return None
        if concept not in self.docs.col_labels: return None
        related = self.spectral.row_named(concept).top_items(10)
        related = [x[0] for x in related if not x[0].endswith('.txt')]
        if concept in self.docs.col_labels:
            documents = [x[1] for x in self.docs.col_named(concept).named_entries()]
        else:
            documents = []

        related_str = ', '.join(related[:5])
        documents_str = ', '.join(documents[:10])
        if len(documents) > 10: documents_str += '...'
        # TODO: make this a jinja template
        html = """
        <h2>%(concept)s</h2>
        <ul>
        <li>Related concepts: %(related_str)s</li>
        <li>From documents: %(documents_str)s</li>
        </ul>
        """ % locals()
        return html

    def save(self, dir):
        def tgt(name): return os.path.join(dir, name)
        def save_pickle(name, obj):
            with open(tgt(name), 'wb') as out:
                pickle.dump(obj, out, -1)

        self.study._step('Saving document matrix...')
        save_pickle("documents.smat", self.docs)

        self.study._step('Saving eigenvectors...')
        save_pickle("spectral.rmat", self.spectral)
        
        self.study._step('Saving projections...')
        save_pickle('projections.dmat', self.projections)

        self.study._step('Saving magnitudes...')
        save_pickle('magnitudes.dvec', self.magnitudes)

        self.study._step('Writing reports...')
        # Save stats
        write_json_to_file(self.stats, tgt("stats.json"))
        self.write_core(tgt("core.txt"))
        self.write_report(tgt("report.html"))

        # Save input contents hash to know if the study has changed.
        save_pickle('input_hash.pickle', self.study.get_contents_hash())

    @classmethod
    def load(cls, dir, for_study):
        def tgt(name): return os.path.join(dir, name)
        def load_pickle(name):
            with open(tgt(name), 'rb') as f:
                return pickle.load(f)

        # Either this will all fail or all succeed.
        try:
            input_hash = load_pickle('input_hash.pickle')
        except IOError:
            raise OutdatedAnalysisError()
        cur_hash = for_study.get_contents_hash()
        if input_hash != cur_hash:
            raise OutdatedAnalysisError()
        
        for_study._step('Loading document matrix...')
        docs = load_pickle("documents.smat")
        for_study._step('Loading eigenvectors...')
        spectral = load_pickle("spectral.rmat")
        for_study._step('Loading projections...')
        projections = load_pickle("projections.dmat")
        for_study._step('Loading magnitudes...')
        magnitudes = load_pickle("magnitudes.dvec")

        # Load stats
        for_study._step('Loading stats...')
        stats = load_json_from_file(tgt("stats.json"))

        return cls(for_study, docs, projections, spectral, magnitudes, stats)

class StudyLoadError(Exception): pass

class StudyDirectory(object):
    '''
    A StudyDirectory manages the directory representing a study. It has three responsibilites:
     - loading the documents, both study and canonical
     - storing settings, such as the number of axes to compute
     - caching analysis results for speed
    '''
    def __init__(self, dir, status_callback=None):
        self.dir = dir.rstrip(os.path.sep)
        self.load_settings()
        self.status_callback = status_callback

    @staticmethod
    def make_new(destdir):
        # make a new study... the hard way.
        def dest_path(x): return os.path.join(destdir, x)
        try:
            os.mkdir(destdir)
            for dir in ['Canonical', 'Documents', 'Matrices', 'Results']:
                os.mkdir(dest_path(dir))
            shutil.copy(os.path.join(package_dir, 'study_skel', 'Matrices', 'conceptnet_en.assoc.smat'),
                        os.path.join(destdir, 'Matrices', 'conceptnet_en.assoc.smat'))
            write_json_to_file({}, dest_path('settings.json'))
        except (IOError, OSError):
            raise StudyLoadError

        return StudyDirectory(destdir)
    
    def _ensure_dir_exists(self, targetdir):
        path = os.path.join(self.dir, targetdir)
        if not os.path.exists(path):
            os.mkdir(path)

    def load_settings(self):
        try:
            self.settings = load_json_from_file(self.get_settings_file())
        except (IOError, ValueError, OSError):
            self.settings = {}

    def save_settings(self):
        write_json_to_file(self.settings, self.get_settings_file())

    def _step(self, msg):
        logger.info(msg)
        if self.status_callback is not None:
            self.status_callback(msg)
    
    def study_path(self, path):
        return self.dir + os.path.sep + path

    def get_settings_file(self):
        return self.study_path("settings.json")

    def get_canonical_dir(self):
        self._ensure_dir_exists("Canonical")
        return self.study_path("Canonical")

    def get_documents_dir(self):
        self._ensure_dir_exists("Documents")
        return self.study_path("Documents")

    def get_matrices_dir(self):
        self._ensure_dir_exists("Matrices")
        return self.study_path("Matrices")
        
    def get_results_dir(self):
        self._ensure_dir_exists("Results")
        return self.study_path("Results")
    
    def listdir(self, dir, text_only, full_names):
        files = os.listdir(self.study_path(dir))
        if text_only: files = [x for x in files if x.endswith('.txt')]
        if full_names:
            return [self.study_path(os.path.join(dir, x)) for x in files]
        else:
            return files

    def convert_old_study(self):
        print "Converting Divisi1 to Divisi2 study."
        from csc.divisi2.network import conceptnet_matrix
        cnet_matrix = conceptnet_matrix('en')
        divisi2.save(cnet_matrix, self.study_path(os.path.join("Matrices",
        "conceptnet_en.smat")))

    def get_matrices_files(self):
        self._ensure_dir_exists("Matrices")
        dirlist = self.listdir('Matrices', text_only=False, full_names=True)
        if ('conceptnet.pickle' in dirlist and
            'conceptnet_en.smat' not in dirlist):
            self.convert_old_study()
            
        return self.listdir('Matrices', text_only=False, full_names=True)

    def get_documents(self):
        study_documents = [Document.from_file(filename, name=os.path.basename(filename))
                           for filename in self.listdir('Documents', text_only=True, full_names=True)]
        return study_documents

    def get_canonical_documents(self):
        self._ensure_dir_exists("Canonical")
        canonical_documents = [CanonicalDocument.from_file(filename, name=os.path.basename(filename))
                               for filename in self.listdir('Canonical', text_only=True, full_names=True)]
        return canonical_documents

    def get_matrices(self):
        return dict((os.path.basename(filename), divisi2.load(filename))
                    for filename in self.get_matrices_files()
                    if filename.endswith('.smat'))
    

    def get_study(self):
        try:
            return Study(name=self.dir.split(os.path.sep)[-1],
                         documents=self.get_documents(),
                         canonical=self.get_canonical_documents(),
                         other_matrices=self.get_matrices(),
                         settings = self.settings
                        )
        except (IOError, OSError):
            raise StudyLoadError

    def analyze(self):
        study = self.get_study()
        results = study.analyze()
        self._ensure_dir_exists('Results')
        results.save(self.study_path('Results'))
        return results

    def set_setting(self, key, value):
        self.settings[key] = value
        self.save_settings()

    def set_num_axes(self, axes):
        self.set_setting('axes', axes)
    
    def get_existing_analysis(self):
        # FIXME: this loads the study twice, I think
        try:
            return StudyResults.load(self.study_path('Results'), self.get_study())
        except OutdatedAnalysisError:
            print "Skipping outdated analysis."
            return None

def run_study(dirname):
    study = StudyDirectory(dirname)
    study.analyze()

def main():
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1:
        run_study(sys.argv[1])
    else:
        print 'Run "luminoso-study StudyDir" to analyze a study directory.'

import sys
if __name__ == '__main__':
    main()

