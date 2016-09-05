from invidx import InvertedIndex
import numpy as np
import os
import pickle



class Ranker():
    def __init__( self, invfile_path, dic_inv ):

        print invfile_path
        print os.path.isfile(invfile_path)
        self.index = InvertedIndex( invfile_path )

        with open(dic_inv, 'rb') as f:
            self.indices=pickle.load( f )

    def build_retrieved_list(self, scores):
        """
        Generate result lists (image names and list with scores)
        """

        res = self.index.rank(scores)
        tmp_res = []
        # keep scores too
        tmp_scores = []

        # build the list
        tmp_res = []
        #print rank, "<--"
        for i, k in res:
            tmp_res.append( self.indices[i] )
            tmp_scores.append( k )


        # compute the difference with the difference
        diff = list(set(self.indices.values())-set(tmp_res))

        # shuffle to fill the rest of the list
        np.random.shuffle(diff)

        scores_diff = np.zeros( (len(diff,)) )

        final = []
        final_scores = []

        final.extend(tmp_res)
        final.extend(diff)

        final_scores.extend(tmp_scores)
        final_scores.extend(scores_diff)

        # remove extension for evaluation
        f = lambda x: x.split('.')[0]
        final = map(f, final)

        return final, final_scores
