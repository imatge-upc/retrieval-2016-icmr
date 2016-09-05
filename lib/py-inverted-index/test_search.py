#!/usr/bin/env python
"""
Example of search
"""
from invidx import InvertedIndex
import numpy as np

index_filename = 'index.dat'
term_to_lookup = 10
query = [(10, 0.5), (15, 0.5)]

# load the index (should be fast, uses mmap)
index = InvertedIndex(index_filename)

# print stats
print 'word count', index.word_count
print 'document count', index.document_count
print 'entry count', index.entry_count

# lookup a document
print '\ntest lookup'
matches = index.lookup(term_to_lookup)
print 'matches', matches

# score documents
print '\ntest score'
scores = index.score(query)
print 'scores', scores

# rank top 10-documents 
print '\ntest rank'
rank = index.rank(query, 10)
print 'ranking', rank