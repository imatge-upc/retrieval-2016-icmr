#!/usr/bin/env python
"""
Example of indexing.
"""
from invidx import InvertedIndex, InvertedIndexBuilder
import numpy as np

index_filename = 'index.dat'
n_docs = 100
n_words = 1000
words_per_doc = 50

indexer = InvertedIndexBuilder(n_words)

# generate some documents to index
for doc in range(n_docs):
    
    # generate random tokens (words)
    words = np.random.randint(0, n_words, words_per_doc)
    words = list(set(words))

    # score words using normalized term frequencies 
    word_counts = np.random.randint(1,10,len(words))
    word_scores = word_counts / float(len(words))

    # create list of tuples
    scores = zip(words, word_scores)

    # insert the document
    indexer.insert(scores)

# print stats
print 'word count', indexer.word_count
print 'document count', indexer.document_count
print 'entry count', indexer.entry_count

# save the index
indexer.save(index_filename)

