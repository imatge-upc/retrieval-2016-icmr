#ifndef INDEX_H_88D4A163
#define INDEX_H_88D4A163

#include "IndexBuilder.h"

class Index { 
public:
    
    struct __attribute__((__packed__)) EntryList {
        uint64_t count;
        IndexEntry* entries; 
    };
    
    // Ctors
    Index();
    Index(const char* filename);
    
    // Disallow copy
    Index(const Index&) = delete;
    Index& operator= (const Index&) = delete;
    
    // Allow move
    Index(Index&&) = default;
    Index& operator= (Index&&) = default;
    
    // Dtor
    ~Index();
    
    // Load/unload
    void load(const char* filename);
    void unload();
    
    // Lookups
    const EntryList* lookup(id_type word) const { return &(index[word]); }
    
    // Info
    size_t entryCount() const { return n_entries; }
    size_t documentCount() const { return n_docs; }
    size_t wordCount() const { return index.size(); }

private:
    
    // Number of documents
    size_t n_docs;
    
    // Number of entries
    size_t n_entries;
    
    // Index
    std::vector< EntryList > index;
    
    // mapped memory
    struct {
        off_t size;
        void* data;
    } mm;
};

#endif /* end of include guard: INDEX_H_88D4A163 */
