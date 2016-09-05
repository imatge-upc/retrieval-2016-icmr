#include "Scorer.h"

#include <cmath>
#include <iostream>
#include <map>

Scorer::Scorer() {}
Scorer::~Scorer() { }

InnerProductScorer::InnerProductScorer() {}
InnerProductScorer::~InnerProductScorer() { }


std::vector<double> InnerProductScorer::score(
    const Index* index, 
    const Query& query)
{
    // Alloc scores
    std::vector<double> scores(index->documentCount(), 0.0);
    
    // Iterate over query terms 
    for (const IndexEntry& term : query) {
        
        // Lookup docs containing term
        const Index::EntryList* docs = index->lookup(term.id);
        
        // For each matching doc, add score product to score
        for (size_t i = 0; i < docs->count; i++) {
            IndexEntry entry = docs->entries[i];
            double score = entry.score * term.score;
            scores[entry.id] += score;
        }
    }
    
    return scores;
}

std::map<id_type, double> InnerProductScorer::score_map(
    const Index* index, 
    const Query& query)
{

    // map from doc id to word score pairs
    std::map<id_type, double> scores;
    
    // Iterate over query terms 
    for (const IndexEntry& term : query) {
        
        // Lookup docs containing term
        const Index::EntryList* docs = index->lookup(term.id);
        
        // For each matching doc, add tf delta to scores
        for (size_t i = 0; i < docs->count; i++) {
            IndexEntry entry = docs->entries[i];
            double score = entry.score * term.score;

            if (scores.find(entry.id) == scores.end()) {
                scores[entry.id] = score;
            } else {
                scores[entry.id] += score;
            }
        }
    }
    
    return scores;
}
