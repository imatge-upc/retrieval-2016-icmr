#ifndef INVERTEDINDEX_HPP_C2BB3155
#define INVERTEDINDEX_HPP_C2BB3155

#include <cstddef>
#include <cstdint>
#include <sys/types.h>
#include <vector>


typedef uint32_t id_type;
typedef float score_type;


/**
 * An entry in the index.
 */
struct __attribute__((__packed__)) IndexEntry {
    id_type id;
    score_type score;
    
    IndexEntry() : id(0), score(0.0f) { }
    IndexEntry(id_type id_, score_type score_) : id(id_), score(score_) { }
};

class IndexBuilder {
public:
    
    typedef std::vector< IndexEntry > EntryList;
    typedef std::vector< EntryList > WordList;
    
    IndexBuilder(size_t nwords, size_t reserve = 0);
    
    size_t addDocument() { 
        return n_documents++; 
    }
    
    size_t addDocument(const EntryList& words) {
        for (IndexEntry entry: words) {
            index[entry.id].push_back(IndexEntry(n_documents, entry.score));
        }
        n_entries += words.size();
        return n_documents++;
    }
    
    void insertWord(id_type word, score_type score) {
        index[word].push_back(IndexEntry(n_documents-1, score));
        n_entries++;
    }

    size_t documentCount() const { return n_documents; }
    size_t entryCount() const { return n_entries; }
    size_t wordCount() const { return index.size(); }
    size_t estimateSizeBytes() const;
    
    void save(const char* filename);
    
private:
    size_t n_entries;
    size_t n_documents;
    WordList index;
};


#endif /* end of include guard: INVERTEDINDEX_HPP_C2BB3155 */
