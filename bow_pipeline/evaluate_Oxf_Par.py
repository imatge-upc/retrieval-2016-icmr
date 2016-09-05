#!/usr/bin/python
import os, glob
import numpy as np

'''
--------------------
OXFORD
--------------------
'''

def store_list( q_frame, final, path_out):
    """
    Store the list (.txt) in the specific path
    """
    if not os.path.exists( path_out ):
        os.makedirs( path_out )

    fid = open( os.path.join(path_out, os.path.basename(q_frame) +'.txt'), "wb" )
    for k in final:
        fid.write( "{}\n".format( k ) )
    fid.close()

def evaluate_results( path_workspace, settings, path_lists=None ):

	# queries
	if settings["dataset"] == "oxford":
		path_gt = r"/media/eva/Eva\ Data/Datasets/Oxford_Buildings/gt_files"
		query_names = ["all_souls", "ashmolean", "balliol","bodleian", "christ_church", "cornmarket","hertford","keble","magdalen","pitt_rivers","radcliffe_camera"]

	elif settings["dataset"] == "paris":
		path_gt = r"/media/eva/Eva\ Data/Datasets/Paris_dataset/gt_data"
		query_names = ["defense", "eiffel","invalides", "louvre", "moulinrouge","museedorsay","notredame","pantheon","pompidou","sacrecoeur", "triomphe"]


	aps = []
	dic_res={}
	for q_name in query_names:
		for i in range(1,6):
			cmd = "./compute_ap {}_{} {}/{}_{}.txt > tmp.txt".format( os.path.join(path_gt, q_name), i, path_lists, q_name, i)
			# execute command
			os.system( cmd )
			# retrieve result
			ap = np.loadtxt("tmp.txt")
			# store in dict
			dic_res[q_name+"_"+str(i)]=ap
			# show screen
			print "{} {} {}".format(ap, q_name, i)
			aps.append(ap)

	print len(aps)

	tot=[]
	for q in query_names:
		ap=[]
		for i in range(1,6):
			ap.append( dic_res[q+"_"+str(i)] )
		tot.append( np.average(ap) )
		print "{}  {}".format(q, np.mean(ap))


	print
	print np.mean(tot), np.mean(aps)
	print  np.mean(aps)



def evaluate_one_frame( frame, path_workspace, settings, path_lists=None ):

    # queries
    if settings["dataset"] == "oxford":
        path_gt = r"/media/eva/Eva\ Data/Datasets/Oxford_Buildings/gt_files"
        query_names = ["all_souls", "ashmolean", "balliol","bodleian", "christ_church", "cornmarket","hertford","keble","magdalen","pitt_rivers","radcliffe_camera"]

    elif settings["dataset"] == "paris":
        path_gt = r"/media/eva/Eva\ Data/Datasets/Paris_dataset/gt_data"
        query_names = ["defense", "eiffel","invalides", "louvre", "moulinrouge","museedorsay","notredame","pantheon","pompidou","sacrecoeur", "triomphe"]


    cmd = "./compute_ap {} {}/{}.txt > tmp.txt".format( os.path.join(path_gt, frame),  path_lists, frame )
    # execute command
    os.system( cmd )
    # retrieve result
    ap = np.loadtxt("tmp.txt", dtype='float32')
    return ap
