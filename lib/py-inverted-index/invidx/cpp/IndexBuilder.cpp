#include "IndexBuilder.h"
#include "IndexWriter.h"

IndexBuilder::IndexBuilder(size_t nwords, size_t reserve) 
: n_entries(0), n_documents(0), index(nwords) 
{
    if (reserve) {
        for (EntryList& entries : index) {
            entries.reserve(reserve);
        }
    } 
}

void IndexBuilder::save(const char* filename) {
    IndexWriter file(filename);
    file.writeHeader(wordCount(), n_entries, n_documents);

    for (size_t i = 0, n = index.size(); i < n; ++i) {
        EntryList& entries = index[i];
        file.writeEntryList(i, entries);
    }
    
    file.writeFooter();
}

size_t IndexBuilder::estimateSizeBytes() const {
    return wordCount() * sizeof(uint64_t) + n_entries * sizeof(IndexEntry);
}

