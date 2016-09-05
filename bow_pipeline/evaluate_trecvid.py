import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import timeit


def store_trecvid_rankings( q_frame, frames, shots, scores, path_out):
    """
    Store the list (.txt) in the specific path
    """
    fid = open( os.path.join( path_out, os.path.basename(q_frame)) +'.txt', "wb" )
    for f,sh,s in zip(frames, shots, scores):
        fid.write("{}\t{}\t{}\n".format( f,sh,s ))
    fid.close()

def build_retrieved_list_trecvid( final, final_scores):
	"""
	build lists of frames, shots and scores for trecvid
	Keep only the highest score per frame.
	"""
	shots = []
	frames_s = []
	shots_score = []
	for frame, score in zip(final, final_scores):
		shot = os.path.basename( frame ).split('_')[0] + '_' + os.path.basename( frame ).split('_')[1]
		if shot not in shots:
			shots.append( shot )
			frames_s.append( frame )
			shots_score.append( score )

	return frames_s, shots, shots_score

def relnotrel( fileGT, id_q, rankingShots ):
    '''
    Function to define if the elements of a ranking list are or not relevant.
    args:
        (fileGT, id_q (namequery), ranking shots)

     - fileGT == string path to the file
     - id_q == string with the query identifier
     - rankingshots == list with the shot names (strings)
    '''

    labelRankingShot = []
    t_shot = []
    a = np.loadtxt( fileGT, dtype='string' )
    q_ids = a[:,0].astype(int)
    # Extract shots for the query
    t_shot = a[ q_ids==int(id_q) ]

    # Extract relevant shots for the query
    t_shot = t_shot[ t_shot[:,3] == '1' ]

    # Total Number of relevant shots in the ground truth
    nRelTot = np.shape( t_shot )[0]
    labelRankingShot = np.zeros((1, len(rankingShots)))

    i = 0
    for shotRanking in rankingShots:
        if shotRanking in t_shot:
            labelRankingShot[0, i ] = 1
        i +=1


    return labelRankingShot, nRelTot



def AveragePrecision( fileGT, id_q, rankingshots, N=None ):
    '''
    Function to compute the average precision of a ranking list
    args:
        (fileGT, id_q (namequery), ranking shots)

     - fileGT == string path to the file
     - id_q == string with the query identifier
     - rankingshots == list with the shot names (strings)
    '''

    (relist, nRelTot) = relnotrel( fileGT, id_q, rankingshots )

    map_ = 0
    accu = 0

    if N is None:
        N = np.shape(relist)[1]

    if N < np.shape(relist)[1]:
        N_limit = N
    else:
        N_limit =  np.shape(relist)[1]
    #print "AP on {}".format(N_limit)
    for k in range( N_limit ):

        if relist[0,k] == 1:
            accu +=1
            map_ += float(accu)/ float(k+1)
    #print map_
    return (map_/nRelTot)
