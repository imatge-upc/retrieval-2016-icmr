from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import pickle
import leveldb
from sklearn.metrics.pairwise import pairwise_distances
from queries import *
from sklearn.metrics.pairwise import pairwise_distances
from IPython import embed
from evaluate_Oxf_Par import *
from evaluate_trecvid import *


# import reader
import sys, os
sys.path.insert(0, '/home/eva/Workspace/icmr_pipeline/bow_pipeline')

from reader import *
from utils import *


'''
--------------------
OXFORD
--------------------
'''
save_pca=True # true when training PCA

# for sumpooling
#dataset_name = "oxford"
#dataset_name = "paris106k"
dataset_name = "paris"
#dataset_name = "oxford105k"
#dataset_name = "trecvid_subset"

pooling='sum'
if pooling == "max":
	whiten=False
else:
	whiten=True

network="vgg16"
list_layers=['pool5']
path_workspace = "/home/eva/Workspace/icmr_pipeline/"
#settings=load_settings("settings.json")
settings=load_settings(os.path.join( path_workspace, "{}/settings.json".format(dataset_name)) )

#dim_input="336_256" #our
#dim_input="586_586" # babenko
#dim_input="672_512" # double ours
dim_input="1024_768" # yannis
#dim_input="384_288" #trecvid
#dim_input="768_576"

# check this!!

#new_dim=(21,16) # ours simple
#new_dim=(42,32) # ours bow
new_dim=(32,24)
#new_dim=(37,37) # babenko
#new_dim=(48,36) # trecvid bow
#new_dim = (24,18) #trecvid pooling
input_size=(1,3, int(dim_input.split('_')[0]), int(dim_input.split('_')[1]) )

path_store_list = os.path.join( path_workspace, dataset_name,"lists_pooling/{}/{}/{}/{}".format( pooling,network,dim_input,list_layers[0] ) )

pca_dim=512
masking=3
augmentation=[0]
QUERY_EXPANSION=False
top=[5, 10]
it=2


path_output=str(settings["path_output"])
path_models=str(settings["path_models"])
pathDB=os.path.join(path_output,'features',network, dim_input)
pca_model_name = os.path.join( path_models, pooling,  network, dim_input, 'pca_{}_{}_{}_{}.pkl'.format(dim_input, list_layers[0], pooling, pca_dim) )
dataset_raw_name = os.path.join( path_output, pooling, network, dim_input, 'raw_dataset_{}_{}.npy'.format(dim_input,list_layers[0] ) )

path_images=str(settings["path_images"])
path_queries=str(settings["path_queries"])

input_size=(1,3, int(dim_input.split('_')[0]),  int(dim_input.split('_')[1]) )
print
print "Reading from: {}".format(pathDB)
print "Layers: {}".format(list_layers)
print "PCA: {}".format(pca_model_name)
print "dataset raw: {}".format(dataset_raw_name)
print "Input network: {}".format(input_size)
print "Path storing lists {}".format(path_store_list)
print
s = raw_input("--> Press key")



def get_dataset( settings, list_layers, pooling, pathDB, pca_model_name=None ):
	"""
	Read pooled features from txt file paths.
	Specify layers to read, pooling (sum/average) and pca model
	"""

	# get keyframes and labels
	keyframes, labels = get_all_keyframes( settings )

	# init reader
	reader = Local_Feature_ReaderDB( pathDB, list_layers, l2norm=True, applyPCA=pca_model_name )

	#loop over keyframes to get l2-features
	i = 1
	i_n = 1
	dataset = []
	for name, label in zip( keyframes, labels ):

		try:
			feature = reader.pool_features( os.path.basename( name ), vertical=label, new_dim=None, pooling=pooling, interpolation=1 )
			dataset.append( feature )
		except:
			fid = open("not_read.txt", "a+")
			fid.write( "{}\t{}\n".format(name, label) )
			fid.close()
			print "not read {} {}".format(i_n, name)
			i_n += 1

		if i%100 == 0:
			print "{}\t{}\t{}\t{}".format( i, os.path.basename(name), label, feature.shape )
		i += 1

	return np.array(dataset), keyframes



class Query_Extractor():
	def __init__(self, settings, networkname, list_layers, in_dim,
		new_dim=None, augmentation=[0], pca_model_name=None ):


		self.dataset = settings["dataset"]
		self.augmentation = augmentation

		if self.dataset == "oxford":
			self.queries = Queries_oxford(str(settings["path_gt_files"]), str(settings["path_images"]))
		elif self.dataset == "paris":
			self.queries = Queries_paris(str(settings["path_gt_files"]), str(settings["path_images"]))
		elif self.dataset == "trecvid_subset":
			self.queries = Queries_trecvid( settings["path_queries"] )


		self.path_images = str(settings["path_queries"])
		# Init local extractor
		self.fx = Local_Extractor.from_settings_for_ranking(settings, networkname, list_layers, in_dim )

		# apply PCA if requiered
		self.pca_model=pca_model_name
		if pca_model_name !=None:
			self.pca_model=load_obj(pca_model_name)

		# new dim
		if new_dim is None:
			self.new_dim=fx.layer_dimensions
		else:
			self.new_dim=new_dim

	def get_pooled_features(self, frame, pooling='sum', mask=None ):
		# check the image extension
		if '.jpg' not in frame and '.png' not in frame:
			frame = frame+'.jpg'

		image = caffe.io.load_image( os.path.join( self.path_images, frame ) )

		# extract features -- default size
		maps = self.fx.get_feature_maps_single_image( image, new_dim=None )

		if pooling=='sum':
			if mask is not None:
				features_ = np.sum( mask*maps[maps.keys()[0]].squeeze(), axis=1 ).sum(axis=1)
			else:
				features_ = np.sum( maps[maps.keys()[0]].squeeze(), axis=1 ).sum(axis=1)

		elif pooling=='max':
			if mask is not None:
				features_ = np.max( mask*maps[maps.keys()[0]].squeeze(), axis=1 ).max(axis=1)
			else:
				features_ = np.max( maps[maps.keys()[0]].squeeze(), axis=1 ).max(axis=1)

		if len(maps.keys()) > 1:
			for layer in maps.keys()[1:]:
				if pooling=='sum':
					if mask is not None:
						features_ = np.concatenate( (features_, np.sum( mask*maps[maps.keys()[0]].squeeze(), axis=1 ).sum(axis=1)) )
					else:
						features_ = np.concatenate( (features_, np.sum( maps[maps.keys()[0]].squeeze(), axis=1 ).sum(axis=1)) )

				if pooling=='max':
					if mask is not None:
						features_ = np.concatenate( (features_, np.max( mask*maps[maps.keys()[0]].squeeze(), axis=1 ).max(axis=1)) )
					else:
						features_ = np.concatenate( (features_, np.max( maps[maps.keys()[0]].squeeze(), axis=1 ).max(axis=1)) )

				print "adding layer {}, total shape {}".format( layer, features_.shape )

		# normalize feature
		features =  normalize( features_ )
		del features_
		print "check features -- ", features.shape, '  ',  np.sqrt( np.sum(features**2) )

		# apply PCA
		if self.pca_model is not None:
			# transform
			features = self.pca_model.transform( features )

			# l2 norm again
			features = normalize( features )

		return features.squeeze()



def get_ranking( dataset, keyframes, qe, list_queries, in_dim, pooling=pooling, masking=None, vertical=True, path_store_list=None):
	# modify input network id required!
	print "masking ", masking
	if not vertical:
		print 'Switching horitzontal!'
		qe.fx.reshape_input(in_dim[0], in_dim[1], in_dim[3], in_dim[2])
		print qe.fx.blobs['data'].data.shape

	# iterate over vertical topic frames
	for i, q_frame in enumerate(list_queries):

		#map q_id
		if qe.dataset == "trecvid_subset":
			# topic id
			topic = os.path.basename( q_frame ).split('.')[0]
			# frame id
			frame_id = os.path.basename( q_frame ).split('.')[1]

			if os.path.exists( os.path.join( path_store_list, "results.txt") ):
				fout = open( os.path.join( path_store_list, "results.txt"), "a+" )
			else:
				fout = open( os.path.join( path_store_list, "results.txt"), "w" )
		else:
			# topic id
			topic = qe.queries.parse_topic_fr[ os.path.basename(q_frame).split('.')[0] ]
			#map topic -- this is different depending on the dataset!
			file_parts=topic.split('_')
			if len(file_parts)==3:
				topic = file_parts[0]+"_"+file_parts[1]
			else:
				topic = file_parts[0]

			frame_id = os.path.basename( q_frame ).split('.')[0]

		# extract mask if requiered
		if masking == 1:
			mask = qe.queries.get_mask_frame( q_frame, dim=qe.new_dim)
			print "Using mask ", mask.shape, np.where(mask.flatten() == 1)[0].shape[0]

		elif masking == 2:
			# check vertical/horitz
			if not vertical:
				dim_ = (qe.new_dim[1], qe.new_dim[0])
			else:
				dim_ = qe.new_dim
			# bbx on fg
			bbx = conver_bbx_mask( qe.queries.get_mask_frame( q_frame, dim=None), dim_ )
			mask = np.zeros( dim_ )
			mask[  bbx[1]:bbx[3], bbx[0]:bbx[2] ]=1
			print "checking square _ ", mask.shape, maks[mask==1].shape

		elif masking == 3:
			mask0 = qe.queries.get_mask_frame( q_frame, dim=None)
			#mask1 = get_weighted_map( mask0, mode='distance', sigma=None )
			mask1 = get_distanceTransform(mask0)

			# check vertical/horitz
			if not vertical:
				dim_ = (qe.new_dim[1], qe.new_dim[0])
			else:
				dim_ = qe.new_dim

			# reshape mask to feature maps dim
			mask = reshape_maps( np.expand_dims(mask1, axis=0 ) , dim_).squeeze()
		elif masking == 0:
			mask=None
			print "No Mask", masking

		# get pooled features
		feature = qe.get_pooled_features( q_frame, pooling=pooling, mask=mask )
		#print "Feature shape ", feature.shape, "  ",  pooling

		# get cosine score to dataset
		scores = pairwise_distances( feature, dataset, metric='cosine' ).squeeze()
		idx = np.argsort( scores )

		# sort final list
		final = np.array(keyframes)[idx]
		final_scores = scores[idx]

		# remove path and extension of the files
		final = [ os.path.basename( k ).split('.')[0] for k in final ]
		#print "{} {} {}".format(topic, q_frame, final[:2])

		# if it is trecvid..
		if qe.dataset == "trecvid_subset":
			frames_s, shots, shots_score = build_retrieved_list_trecvid( final, final_scores )

			# store lists
			store_trecvid_rankings( topic+'_'+frame_id, frames_s, shots, shots_score, path_out=path_store_list )

			ap = AveragePrecision( "/media/eva/Eva Data/Datasets/ins15/8_groundtruth/src/ins.search.qrels.tv13", topic, shots )
			print "{}\t{}\t{}\t{} ".format( i, topic, os.path.basename(frame_id), ap )
			fout.write( "{}\t{}\t{}\t{}\n".format( i, topic, os.path.basename(frame_id), ap ) )
			fout.close()
		# store lists
		else:
			store_list( qe.queries.parse_topic_fr[frame_id], final, path_out=path_store_list )

	print "DONE {}".format( vertical )



def main():

	# rm prev results
	files = glob.glob( os.path.join( path_store_list, "*" ) )
	if len(files) > 0:
		for f in files:
			os.remove(f)
	if not os.path.exists(path_store_list):
		os.makedirs( path_store_list )
	# load raw
	# LOAD DATASET ------------------------------------------
	if not os.path.isfile( dataset_raw_name ):
		print "Loading ... "
		dataset, keyframes = get_dataset( settings, list_layers, pooling, pathDB )
		print "Loaded {}".format( dataset.shape )
		check_path_file(dataset_raw_name, create_if_missing=True)
		save_obj(dataset, dataset_raw_name)

	else:
		print "Loading numpy ... "
		dataset = load_obj( dataset_raw_name )
		keyframes, labels = get_all_keyframes( settings )


	print "Raw dataset {}".format( dataset.shape )
	s = raw_input("--> Press key")

	# LOAD PCA -----------------------------------------------
	if not os.path.isfile( pca_model_name ):
		print "reading...."
		# APPLY DATASET
		# check if pca model exists
		if not os.path.isfile( pca_model_name ):
			# if not compute it and store it
			print "Training PCA"
			# get Dataset
			pca_model = PCA(n_components=pca_dim, whiten=whiten)
			dataset = pca_model.fit_transform( dataset )

			if save_pca:
				check_path_file(pca_model_name, create_if_missing=True)
				save_obj(pca_model, pca_model_name)

	else:
		print "reading models...."
		pca_model = load_obj( pca_model_name )
		dataset = pca_model.transform( dataset )
		keyframes, labels = get_all_keyframes( settings )


	print "Loaded {} with PCA {}".format( dataset.shape, pca_model_name )
	print "masking {}, augmentation {}, pca_model_name {}, new_dim {}, input_size {}, pooling {}".format( masking, augmentation,
																										 pca_model_name, new_dim, input_size, pooling)
	s = raw_input("--> Press key")


	# GET KEYFRAMES queries -----------------------------------------------------------------------------------

	keyframes_q, labels_q = create_list_queries(settings)
	qe = Query_Extractor( settings, network, list_layers, input_size,
							 new_dim=new_dim, augmentation=augmentation, pca_model_name=pca_model_name )

	get_ranking( dataset, keyframes, qe, keyframes_q[ labels_q==True ] , in_dim=input_size, pooling=pooling, masking=masking, vertical=True, path_store_list=path_store_list)
	get_ranking( dataset, keyframes, qe, keyframes_q[ labels_q==False ], in_dim=input_size, pooling=pooling, masking=masking, vertical=False, path_store_list=path_store_list)




if __name__=="__main__":
	main()

	if settings[ "dataset" ] == "trecvid_subset":
		data = np.loadtxt( open(os.path.join( path_store_list, "results.txt" ), 'r' ), dtype='str' )[:,3].astype(np.float32)
		print "Final mAP"
		print np.average( data )

	else:
		evaluate_results( path_workspace, settings, path_lists=path_store_list )
