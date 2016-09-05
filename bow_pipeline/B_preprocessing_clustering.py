import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import caffe
import pickle
from vlfeat import kmeans
from utils import *
from reader import *
import time

log = set_logger("B_preprocessing", level_logger="debug",format_logger="%(asctime)s- %(name)s - %(levelname)s - %(message)s")

"""
Functions
"""
def load_features(keyframes, labels, reader, N=3000000):

	# get dimension of features
	dim=reader.get_dimension
	# total no features per image
	nf_images=dim[1]*dim[2]
	log.debug( "\n{} dim\nnf_images {}\n".format( dim, nf_images ) )
	#s = raw_input("--> Press key")

	# get default total images for the dataset
	total_features=len(keyframes)*nf_images

	# if total features < N, perfect; we just read all!
	if total_features < N:
		log.info("Not enough images. We'll train in {} local features.".format(total_features))
		n_sample_image=None
	else:
		# we sample n_featues across all images -- (if dataset is too big, requiered previous keyframe selection!)
		n_sample_image=int(N/len(keyframes))
		log.info( "{} images.".format( n_sample_image ) )
		if N%len(keyframes) !=0:
			n_sample_image+=1
			log.info("Sampling {} features per image.".format(n_sample_image))
			log.info("{}; total size in RAM {} Gbs".format(N, 4.0*float(n_sample_image)*float(len(keyframes))*float(dim[0])/(1024.0**3.0)))

	features=[]
	# Loading...
	i = 0
	for keyframe, label in zip(keyframes, labels):
		# getting local features
		features.extend( reader.get_features( os.path.basename(keyframe),vertical=label, n_samples=n_sample_image  ) )
		i += 1

		if i%100==0:
			log.info("\t\t{} images \t{} features\t{} Gb".format( i+1, len(features), 4.0*float(len(features))*float(dim[0])/(1024.0**3.0)))

	# convert list into numpy array
	features=np.array(features[:N])
	log.info("\t\t{} images \t{} features\t{} Gb".format( i+1, len(features), 4.0*float(len(features))*float(dim[0])/(1024.0**3.0)))

	log.info("\n feat_dims={}\nDONE\n------".format(features.shape))

	return features


def trainingPCA(features, n_components=256, whiten=True, pca_model_name=None):
	print 'loaded features! {}'.format(features.shape)
	print np.sqrt(sum(features[0,:]**2))

	#print 'Features l2 normalization'
	#features = normalize(features)
	#print np.sqrt(sum(features[0,:]**2))

	print 'Feature PCA-whitenning'
	pca_model = PCA(n_components=n_components, whiten=whiten)
	features = pca_model.fit_transform(features)
	print np.sqrt(sum(features[0,:]**2))

	print 'Features l2 normalization'
	features = normalize(features)
	print np.sqrt(sum(features[0,:]**2))

	if pca_model_name is not None:
		print 'saving model...'
		check_path_file(pca_model_name, create_if_missing=True)
		save_obj(pca_model, pca_model_name)

	print 'done! {}'.format(pca_model_name)

	return pca_model



"""
PARAMETERS
"""

TRAIN_PCA=True
TRAIN_CENTROIDS=True
l2norm=True
n_centers=25000
pca_dim=512

'''
Paths
'''
# oxford
#settings=load_settings("/home/eva/Workspace/icmr_pipeline/oxford/settings.json")
#settings=load_settings("/home/eva/Workspace/icmr_pipeline/oxford105k/settings.json")


# paris
settings = load_settings("/home/eva/Workspace/icmr_pipeline/paris/settings.json")

#trecvid
#settings = load_settings("/home/eva/Workspace/icmr_pipeline/trecvid_subset/settings.json")

#dim_input="384_288" #trecvid
dim_input="336_256" # normal oxford
#dim_input="672_512"
#dim_input="586_586"


network="vgg16"
path_output=str(settings["path_output"])
pathDB=os.path.join(path_output,'features',network, dim_input)
list_layers=[ 'conv5_1' ]

pca_model_name =os.path.join(path_output, 'pca',network, dim_input, '---.pkl')
#pca_model_name=None
model_centroids=os.path.join(path_output,'centroids', network, dim_input,'---.pkl')


keyframes, labels = get_all_keyframes( settings )
keyframes = np.array(keyframes)
labels = np.array(labels)
idx = np.arange(keyframes.shape[0])
np.random.shuffle(idx)
#keyframes[idx[:10000]]

#N=1701168
N=3000000 # local feartures

print
print "Reading from: {}".format(pathDB)
print "Layers: {}".format(list_layers)
print "PCA: {}".format(pca_model_name)
print "\t PCA dim={}".format(pca_dim)
print "Centroids: {}".format(model_centroids)
print "\t centroids #={}".format(n_centers)
print
s = raw_input("--> Press key")


if __name__ == "__main__":

	# init the reader
	# get dimensions layer
	print len(keyframes)

	if TRAIN_PCA:

		reader = Local_Feature_ReaderDB( pathDB, list_layers)
		features=load_features(keyframes[idx[:10000]], labels[idx[:10000]], reader, N=N)
		#features=load_features(keyframes, labels, reader, N=1701168)
		del reader
		print 'Features loaded! {} {}'.format(features.shape, (features.shape[0]*features.shape[0]*4 / (1024.**3)))
		print 'Training PCA...'
		t0=time.time()
		pca_model = trainingPCA(features, n_components=pca_dim, whiten=True, pca_model_name=pca_model_name)
		t1=time.time()
		print 'Model created! {}\t elapsed time {}'.format( pca_model_name, t1-t0 )
		del features
	else:
		if pca_model_name!=None:
			pca_model = load_obj( pca_model_name )
		else:
			print pca_model_name, " don't exists"
		print "Computing centroids..."
	if TRAIN_CENTROIDS:

		np.random.shuffle(idx)

		reader = Local_Feature_ReaderDB( pathDB, list_layers, applyPCA=pca_model_name)
		features=load_features(keyframes[idx[:10000]], labels[idx[:10000]], reader, N=N)
		#features=load_features(keyframes, labels, reader, N=1701168)
		del reader
		print "Clustering..."
		t0=time.time()
		clustering_model = kmeans.KMeans( num_centers = n_centers, algorithm='ann', initialization='random' )
		clustering_model.fit(features)
		centers = clustering_model.centers.copy()
		t1=time.time()
		print 'Trained centroids {}. \nElapsed time={}'.format(centers.shape, t1-t0)
		t0=time.time()
		check_path_file(model_centroids)
		save_obj(centers, model_centroids)
		t1=time.time()
		print 'Stored in centroids {}\t elapsed time {}'.format(model_centroids, t1-t0)
