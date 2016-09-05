#ifndef INDEXFILE_H_04FF99CB
#define INDEXFILE_H_04FF99CB

#include "IndexBuilder.h"
#include <cstdint>
#include <cstdio>

#define FOURCC(a,b,c,d) ((uint32_t) (((d)<<24) | ((c)<<16) | ((b)<<8) | (a)))
#define INDEX_FILE_HEADER_MAGIC FOURCC('I','I','D','X')

struct __attribute__((__packed__)) IndexFileHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t word_count;
    uint64_t entry_count;
    uint64_t document_count;
};

class IndexWriter {
public:
    IndexWriter();
    IndexWriter(const char* filename);
    ~IndexWriter();
    
    bool open(const char* filename);
    void close();
    
    void writeHeader(size_t word_count, size_t entry_count, size_t doc_count);
    void writeEntryList(id_type word, const IndexBuilder::EntryList& entries);
    void writeFooter();
    

private:
    FILE* f;
};

#endif /* end of include guard: INDEXFILE_H_04FF99CB */
