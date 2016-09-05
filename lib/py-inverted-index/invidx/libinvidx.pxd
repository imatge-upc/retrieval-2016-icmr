from libc.stdint cimport uint32_t, uint64_t
from libcpp.vector cimport vector
from libcpp.map cimport map

cdef extern from "cpp/Index.h":

    ctypedef uint32_t id_type
    ctypedef float score_type

    cdef cppclass IndexEntry:
        id_type id
        score_type score
        IndexEntry()
        IndexEntry(id_type, score_type)

    cdef cppclass IndexBuilder:
        IndexBuilder(size_t nwords)
        IndexBuilder(size_t nwords, size_t reserve)
        size_t addDocument()
        void insertWord(id_type word, score_type score)
        size_t documentCount()
        size_t entryCount()
        size_t wordCount()
        size_t estimateSizeBytes()
        void save(const char* filename)

    cdef cppclass Index:
        cppclass EntryList:
            uint64_t count
            IndexEntry* entries
        Index()
        Index(const char* filename)
        Index(const Index&)
        void load(const char* filename)
        void unload()
        const EntryList* lookup(id_type word)
        size_t entryCount()  
        size_t documentCount() 
        size_t wordCount()

cdef extern from "cpp/Scorer.h":

    ctypedef vector[IndexEntry] Query
    
    cdef cppclass Scorer:
        Scorer()
        vector[double] score(const Index* index, const Query& query)
        map[id_type, double] score_map(const Index* index, const Query& query)

    cdef cppclass InnerProductScorer(Scorer):
        InnerProductScorer()
