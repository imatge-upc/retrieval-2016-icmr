#include "IndexWriter.h"

//#include <stdio.h>


IndexWriter::IndexWriter() : f(NULL) {
    
}

IndexWriter::IndexWriter(const char* filename) : f(NULL) {
    open(filename);
}

IndexWriter::~IndexWriter() {
    close();
}

bool IndexWriter::open(const char* filename) {
    close();
    f = fopen(filename, "w");
    return static_cast<bool>(f);
}

void IndexWriter::close() {
    if (f) {
        fclose(f);
        f = NULL;
    }
}

void IndexWriter::writeHeader(
    size_t word_count, size_t entry_count, size_t doc_count
) {
    IndexFileHeader hdr;
    hdr.magic = INDEX_FILE_HEADER_MAGIC;
    hdr.version = 0x1;
    hdr.word_count = word_count;
    hdr.entry_count = entry_count;
    hdr.document_count = doc_count;
    fwrite(&hdr, sizeof(IndexFileHeader), 1, f);
}

void IndexWriter::writeEntryList(
    id_type word, const IndexBuilder::EntryList& entries
) {
    uint64_t n = (uint64_t) entries.size();
    fwrite(&n, sizeof(uint64_t), 1, f);
    fwrite(&entries[0], sizeof(IndexEntry), n, f);
}

void IndexWriter::writeFooter() {
}
