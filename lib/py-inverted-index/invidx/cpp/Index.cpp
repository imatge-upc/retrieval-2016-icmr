#include "Index.h"

#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cassert>

#include "IndexWriter.h"

static void* mmap_file(const char* filename, off_t* size) {
    struct stat buf;
    stat(filename, &buf);
    *size = buf.st_size;
    
    // open file
    int fd = open(filename, O_RDONLY, (mode_t)0400);
    
    // mmap the whole file
    void* data = mmap(0, *size, PROT_READ, MAP_SHARED, fd, 0);
    
    // close file
    close(fd);

    // return the data
    return data;
} 

struct MemoryReader {
    MemoryReader(void* mem) : ptr(reinterpret_cast<uint8_t*>(mem)) { }
    
    template <class Type>
    Type* read(size_t n = 1) {
        Type* obj = reinterpret_cast<Type*>(ptr);
        ptr += sizeof(Type) * n;
        return obj;
    }
  
private:
    uint8_t* ptr;  
};

Index::Index() : n_docs(0), n_entries(0), mm({0, NULL}) { 

}

Index::Index(const char* fn) :  n_docs(0), n_entries(0), mm({0, 0}) {
    load(fn);
}

Index::~Index() {
    unload();
}

void Index::load(const char* filename) {
    
    // mmap the file
    mm.data = mmap_file(filename, &mm.size);
    MemoryReader reader(mm.data);
    
    // load the header
    IndexFileHeader* hdr = reader.read<IndexFileHeader>();
    n_docs = hdr->document_count;
    n_entries = hdr->entry_count;
    
    // alloc memory for the index
    index.resize(hdr->word_count);
    
    // Populate index with pointers into the mapped memory
    size_t entries_read = 0;
    for (size_t i = 0; i < index.size(); ++i) {
        EntryList& entry_list = index[i];
        entry_list.count = *(reader.read<uint64_t>());
        entry_list.entries = reader.read<IndexEntry>(entry_list.count);
        entries_read += entry_list.count;
    }
    
    // Ensure we've read the right amount
    assert(entries_read == hdr->entry_count);
}

void Index::unload() {
    if (mm.data) {
        munmap(mm.data, mm.size);
        mm = {0, NULL};
    }
    n_docs = 0;
    n_entries = 0;
    index.clear();
}