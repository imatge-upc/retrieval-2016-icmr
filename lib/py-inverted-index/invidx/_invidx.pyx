#cython: embedsignatures=True
cimport cython

from libcpp.vector cimport vector
from libcpp.map cimport map

import numpy as np

cimport numpy as np

from libinvidx cimport (
    id_type,
    score_type,
    Index, 
    Scorer,
    InnerProductScorer, 
    IndexEntry, 
    IndexBuilder
)

np.import_array()

cdef class InvertedIndexBuilder:
    """
    Build fast inverted indices.

    Example of use with genism's tfidf transformer:

    >>> tfidf = gensim.models.TfidfModel.load('tfidf.model')
    >>> indexer = InvertedIndexBuilder(n_words)
    >>> for doc in corpus:
    >>>     indexer.insert(tfidf[doc])
    >>> indexer.save('index.dat')
    """

    cdef IndexBuilder* builder

    def __cinit__(self, size_t n_words):
        """
        Initialize the index builder.

        Parameters
        ----------
        n_words: int
            The number of words in the vocabulary
        """
        self.builder = new IndexBuilder(n_words)

    def __dealloc__(self):
        """
        Free memory
        """
        del self.builder

    def insert(self, scores):
        """
        Insert a document.

        Parameters
        ----------
        scores: list
            A list of word, score pairs

        Returns
        -------
        doc_id: int
            The document id for the inserted document
        """
        doc_id = self.builder.addDocument()
        for word, score in scores:
            self.builder.insertWord(word, score)
        return doc_id

    def save(self, filename):
        """
        Save the inverted index to a file
        """
        self.builder.save(filename)

    @property
    def entry_count(self):
        """
        Number of entries in the index
        """
        return self.builder.entryCount()

    @property
    def document_count(self):
        """
        Number of documents in the index
        """
        return self.builder.documentCount()

    @property
    def word_count(self):
        """
        Number of words in the index
        """
        return self.builder.wordCount()


cdef class InvertedIndex:
    cdef Index* index
    cdef Scorer* scorer

    def __cinit__(self, filename=None):
        """
        Create a new inverted index

        Parameters
        ----------
        filename: str
            If not None, the index is loaded from the given filename
        """
        self.index = new Index()
        if filename:
            self.load(filename)
        self.scorer = new InnerProductScorer()

    def __dealloc__(self):
        """
        Free memory
        """
        del self.index
        del self.scorer

    def load(self, filename):
        """
        Load the index from a file
        """
        self.index.load(filename)

    def close(self):
        """
        Unload the index (free memory)
        """
        self.index.unload()

    def lookup(self, term):
        """
        Lookup a term and return a list of (doc_id, score) pairs for the term.

        Parameters
        ----------
        term: int
            A single term id

        Returns
        -------
        results: list
            (doc_id, score) pairs
        """
        cdef const Index.EntryList* entryList
        cdef int i = 0
        cdef list words = []

        # lookup term
        entryList = self.index.lookup(term)

        # convert result to list of tuples
        for i in range(entryList.count):
            words.append((entryList.entries[i].id, entryList.entries[i].score))
        return words

    def score(self, query):
        """
        Score a query

        Parameters
        ----------
        query: list
            A list of (term, score) pairs

        Returns
        -------
        results: list
            A list of (doc, score) pairs
        """
        cdef vector[IndexEntry] query_entries
        cdef map[id_type, double] scores
        
        # convert query to query
        for term, score in query:
            query_entries.push_back(IndexEntry(term, score))

        # score query
        scores = self.scorer.score_map(self.index, query_entries)
        return scores

    def rank(self, query, n=None):
        """
        Score and rank results
        """
        scores = self.score(query)
        items = scores.items()
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:n]

    @property
    def entry_count(self):
        """
        Number of entries in the index
        """
        return self.index.entryCount()

    @property
    def document_count(self):
        """
        Number of documents in the index
        """
        return self.index.documentCount()

    @property
    def word_count(self):
        """
        Number of words in the index
        """
        return self.index.wordCount()
    
    



