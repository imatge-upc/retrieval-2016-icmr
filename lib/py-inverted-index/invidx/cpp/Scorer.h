#ifndef SCORER_H_80A3930B
#define SCORER_H_80A3930B

#include <vector>
#include <map>

#include "IndexBuilder.h"
#include "Index.h"

typedef std::vector< IndexEntry > Query;

class Scorer {
public:
    Scorer();
    virtual ~Scorer();
    
    virtual std::vector<double> score(
        const Index* index, 
        const Query& query) = 0;

    virtual std::map<id_type, double> score_map(
        const Index* index, 
        const Query& query) = 0;
};

class InnerProductScorer : public Scorer {
public:
    InnerProductScorer();
    virtual ~InnerProductScorer();
    
    virtual std::vector<double> score(
        const Index* index, 
        const Query& query);

    virtual std::map<id_type, double> score_map(
        const Index* index, 
        const Query& query);
};



#endif /* end of include guard: SCORER_H_80A3930B */
