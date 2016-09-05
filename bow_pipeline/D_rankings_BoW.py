import leveldb
from IPython import embed
from evaluate_Oxf_Par import *
from evaluate_trecvid import *

# import reader
import sys, os
sys.path.insert(0, '/home/eva/Workspace/icmr_pipeline/bow_pipeline')

from C_bow_representation import GPUCodebook
from reader import *
from ranker import *
from utils import *
from queries import *




class Query_Extractor():
	def __init__(self, model_centroids, settings, networkname, list_layers, in_dim,
		new_dim=None, masking=1, augmentation=[0], pca_model_name=None ):

		# Load Centroids and init codebook
		self.centroids = load_obj(model_centroids)
		self.codebook=GPUCodebook(self.centroids)
		self.augmentation=augmentation
		self.dataset =  settings[ "dataset" ]

		if self.dataset == "oxford":
			self.queries = Queries_oxford(str(settings["path_gt_files"]), str(settings["path_images"]))
		elif self.dataset == "paris":
			self.queries = Queries_paris(str(settings["path_gt_files"]), str(settings["path_images"]))
		elif self.dataset == "trecvid_subset":
			self.queries = Queries_trecvid( settings["path_queries"] )

		self.default_input = in_dim
		self.path_images=str(settings["path_images"])
		# Init local extractor
		self.path_gt=str(settings["path_queries"])
		self.fx = Local_Extractor.from_settings_for_ranking(settings, networkname, list_layers, in_dim )

		# apply PCA if requiered
		self.pca_model=None
		if pca_model_name !=None:
			self.pca_model=load_obj(pca_model_name)

		# new dim
		if new_dim is None:
			self.new_dim=fx.layer_dimensions
		else:
			self.new_dim=new_dim
			self.masking=masking

	def scores_for_invertedFile(self, words):
		"""
		Generate the scores for the inverted file.
		arg:	words: assignmets from local features
		ret:	l2-norm scores, only returns elements !=0 in tuple form
		"""
		hist_q = get_bow_hist(words, self.centroids.shape[0])
		idx = np.where( hist_q>0)[0]
		scores = zip(idx,hist_q[idx])
		return scores

	def get_assignments_image(self, image ):
		# switch input if necessary
		self.switching_default_input( image )

		# extract features
		features = self.fx.get_features_single_image( image, new_dim=self.new_dim, pca_model=self.pca_model)

		# compute assignments for image
		assignments = self.codebook.get_assignments(features)
		return assignments

	def get_assignments(self, frame, save=False ):
		"""
		frame is the abs path to the image
		"""
		# load image
		scores = None
		image = caffe.io.load_image(frame)

		# get assignments
		assingments = self.get_assignments_image(image)


		#print mask.shape, image.shape, assingments.shape
		# make sure binary mask -- prob not necessary


		# get words from fg
		if self.masking  ==  1:
			print "masking 1"
			#mask = get_mask(self.queries, self.path_gt, frame, image, self.new_dim)
			mask = self.queries.get_mask_frame( frame, dim=self.new_dim)
			mask[ mask>0.5 ] = 1.0
			mask[ mask<=0.5 ] = 0.0
			assingments=assingments[mask.flatten()>0]
			#print mask.shape, image.shape, assingments.shape

		# get words from bg
		elif self.masking == 2:
			print "masking 2"
			# check vertical/horitz
			dim_ = check_verHor( image,  self.new_dim )
			mask = self.queries.get_mask_frame( frame, dim=None)
			mask[ mask>0.5 ] = 1.0
			mask[ mask<=0.5 ] = 0.0
			# bbx on fg
			bbx = conver_bbx_mask( mask, dim_ )
			mask = np.zeros( dim_ )
			mask[  bbx[1]:bbx[3], bbx[0]:bbx[2] ]=1
			#print "checking square _ ", mask.shape,  mask[mask==1].shape, assingments.shape, assingments[mask.flatten()>0].shape
			assingments=assingments[mask.flatten()>0]

		# get weighted words
		elif self.masking==3:
			print "masking 3"

			# get original mask
			mask = self.queries.get_mask_frame( frame, dim=None)
			mask[ mask>0.5 ] = 1.0
			mask[ mask<=0.5 ] = 0.0
			#mask = get_weighted_map( mask, mode='distance', sigma=None )
			mask = get_distanceTransform(mask)
			# check vertical/horitz
			dim_ = check_verHor( image,  self.new_dim )

			# reshape mask to feature maps dim
			mask = reshape_maps( np.expand_dims(mask, axis=0 ) , dim_).squeeze()

			# get scores for the inv file
			scores = weight_assignments( assingments, mask.flatten(), K=self.centroids.shape[0] )
			#print mask.shape, image.shape, assingments.shape, scores.shape


		# save for amaia
		if save:
			query_name_store = []
			assignments_store = []
			masks_store = []

			query_name_store.append( 0 )
			assignments_store.append( assingments )
			masks_store.append( mask  )

		# add words from augmented images
		if len(self.augmentation)>0:
			print "augmentation"

			tmp = []
			tmp_scores = []

			image_flip = None
			# for each augmentation...
			for k, aug in enumerate(self.augmentation):
				if aug == 0:
					tmp.extend(assingments)
					if scores is not None:
						tmp_scores.extend( scores )
					print 'adding default {}'.format( len(tmp) )
				# AUGMENTATION 1
				if aug == 1:
					print "FLIPPING", len(tmp)
					image_flip=flipImage(image)
					assingments=self.get_assignments_image(image_flip)

					if self.masking==1:
						mask = self.queries.get_mask_frame( frame, dim=self.new_dim)
						mask = flipImage(mask)
						mask[ mask>0.5 ] = 1.0
						mask[ mask<=0.5 ] = 0.0
						assingments=assingments[mask.flatten()>0]

					elif self.masking==2:
						print "masking 2"
						# check vertical/horitz
						dim_ = check_verHor( image,  self.new_dim )
						mask = self.queries.get_mask_frame( frame, dim=None)
						mask[ mask>0.5 ] = 1.0
						mask[ mask<=0.5 ] = 0.0

						# flipp mask
						mask = flipImage(mask)
						# bbx on fg
						bbx = conver_bbx_mask( mask, dim_ )
						mask = np.zeros( dim_ )
						mask[  bbx[1]:bbx[3], bbx[0]:bbx[2] ]=1
						#print "checking square flp_ ", mask.shape,  mask[mask==1].shape, assingments.shape, assingments[mask.flatten()>0].shape
						assingments=assingments[mask.flatten()>0]

					elif self.masking==3:
						# not completed!
						mask = self.queries.get_mask_frame( frame, dim=None)
						mask=flipImage(mask)
						#mask = get_weighted_map( mask )
						mask = get_distanceTransform(mask)

						dim_ = check_verHor( image,  self.new_dim )
						mask = reshape_maps( np.expand_dims(mask, axis=0 ) , dim_).squeeze()
						scores = weight_assignments( assingments, mask.flatten(), K=self.centroids.shape[0] )

					if save:
						query_name_store.append( 1 )
						assignments_store.append( assingments )
						masks_store.append( mask  )

					tmp.extend(assingments)
					if scores is not None:
						tmp_scores.extend( scores )
					print 'adding fp {}'.format( len(tmp) )

				# AUGMENTATION 2
				if aug == 2:
					print "ZOOM", len(tmp)
					crop = get_center_crop( image, percentage=0.5 )
					#crop_mask = get_center_crop( mask, percentage=0.5 )
					assingments=self.get_assignments_image(crop)

					if self.masking==1:
						mask = self.queries.get_mask_frame( frame, dim=self.new_dim)
						mask = get_center_crop( mask, percentage=0.5 )
						mask[ mask>0.5 ] = 1.0
						mask[ mask<=0.5 ] = 0.0
						assingments=assingments[mask.flatten()>0]

					elif self.masking==2:
						print "masking 2"
						# check vertical/horitz
						dim_ = check_verHor( image,  self.new_dim )
						mask = self.queries.get_mask_frame( frame, dim=None)
						mask = get_center_crop( mask, percentage=0.5 )
						mask[ mask>0.5 ] = 1.0
						mask[ mask<=0.5 ] = 0.0

						# bbx on fg
						bbx = conver_bbx_mask( mask, dim_ )
						mask = np.zeros( dim_ )
						mask[  bbx[1]:bbx[3], bbx[0]:bbx[2] ]=1
						#print "checking square flp_ ", mask.shape,  mask[mask==1].shape, assingments.shape, assingments[mask.flatten()>0].shape
						assingments=assingments[mask.flatten()>0]

					elif self.masking==3:
						# not completed!
						mask = self.queries.get_mask_frame( frame, dim=None)
						#mask = get_weighted_map( mask )
						mask = get_distanceTransform(mask)

						mask = get_center_crop( mask, percentage=0.5 )

						dim_ = check_verHor( image,  self.new_dim )
						mask = reshape_maps( np.expand_dims(mask, axis=0 ) , dim_).squeeze()
						scores = weight_assignments( assingments, mask.flatten(), K=self.centroids.shape[0] )

					if save:
						query_name_store.append( 2 )
						assignments_store.append( assingments )
						masks_store.append( mask  )

					tmp.extend(assingments)
					if scores is not None:
						tmp_scores.extend( scores )
					print 'adding zoom {}'.format( len(tmp) )

				# AUGMENTATION 3
				if  aug == 3:
					print "flipped_crop ", len(tmp)

					flipped_crop = get_center_crop( flipImage(image), percentage=0.5 )

					assingments=self.get_assignments_image(flipped_crop)

					if self.masking==1:
						mask = self.queries.get_mask_frame( frame, dim=self.new_dim)
						mask = get_center_crop( flipImage(mask), percentage=0.5 )
						mask[ mask>0.5 ] = 1.0
						mask[ mask<=0.5 ] = 0.0
						assingments=assingments[mask.flatten()>0]

					elif self.masking==2:
						print "masking 2"
						# check vertical/horitz
						dim_ = check_verHor( image,  self.new_dim )
						mask = self.queries.get_mask_frame( frame, dim=None)
						mask = get_center_crop( flipImage(mask), percentage=0.5 )
						mask[ mask>0.5 ] = 1.0
						mask[ mask<=0.5 ] = 0.0

						# bbx on fg
						bbx = conver_bbx_mask( mask, dim_ )
						mask = np.zeros( dim_ )
						mask[  bbx[1]:bbx[3], bbx[0]:bbx[2] ]=1
						#print "checking square flp_ ", mask.shape,  mask[mask==1].shape, assingments.shape, assingments[mask.flatten()>0].shape
						assingments=assingments[mask.flatten()>0]

					elif self.masking==3:
						# not completed!
						mask = self.queries.get_mask_frame( frame, dim=None)
						#mask = get_weighted_map( mask )
						mask = get_distanceTransform(mask)

						mask = get_center_crop( flipImage(mask), percentage=0.5 )

						dim_ = check_verHor( image,  self.new_dim )
						mask = reshape_maps( np.expand_dims(mask, axis=0 ) , dim_).squeeze()
						scores = weight_assignments( assingments, mask.flatten(), K=self.centroids.shape[0] )
					if save:
						query_name_store.append( 3 )
						assignments_store.append( assingments )
						masks_store.append( mask  )

					tmp.extend(assingments)
					if scores is not None:
						tmp_scores.extend( scores )
					print 'adding flipped zoom {}'.format( len(tmp) )

				# save it for amaia
				if save:
					if not os.path.isdir( 'query_store/' ):
						os.mkdir( 'query_store/' )
						fid = open( os.path.join('query_store', frame+'.npy'), 'wb' )
						pickle.dump( query_name_store, fid, )
						pickle.dump( assignments_store, fid )
						pickle.dump( masks_store, fid )
						fid.close()


			assingments=np.array(tmp)
			if scores is not None:
				scores = np.array(tmp_scores)

		return assingments, scores

	def get_histograms(self, frame ):
		""" Get Histograms for an image """

		# dimension centroids
		K=self.centroids.shape[0]

		# compute assignmets -- considering data augmentation/ masking
		assignments, scores = self.get_assignments( frame )

		if scores == None:
			# return the histogram
			return get_bow_hist(assignments, K )

	def switching_default_input(self, ima):
		in_dim = self.default_input

		if ima.shape[0]< ima.shape[1] and in_dim[2]>in_dim[3]:
			print 'Switching horitzontal!'
			self.fx.reshape_input(in_dim[0], in_dim[1], in_dim[3], in_dim[2])

			# switch default
			self.default_input = (in_dim[0], in_dim[1], in_dim[3], in_dim[2])
			print self.fx.blobs['data'].data.shape
			print self.default_input

		if ima.shape[0]> ima.shape[1] and in_dim[2]<in_dim[3]:
			print 'Switching Vertical!'
			self.fx.reshape_input(in_dim[0], in_dim[1], in_dim[3], in_dim[2])

			# switch default
			self.default_input = (in_dim[0], in_dim[1], in_dim[3], in_dim[2])
			print self.fx.blobs['data'].data.shape
			print self.default_input

	def restore_input(self):
		in_dim = self.default_input
		if self.fx.blobs['data'].data.shape != in_dim:
			self.fx.reshape_input(in_dim[0], in_dim[1], in_dim[2], in_dim[3])
			print 'Input restored!'
			print self.fx.blobs['data'].data.shape


def get_ranking( ranker, qe, keyframe  ):
	# get assignments
	ima = caffe.io.load_image( keyframe )

	# switch input if necessary
	qe.switching_default_input( ima )

	# visual words
	fout = open( "tmp_words.txt", "a+" )
	words, scores = qe.get_assignments( keyframe )
	fout.write( "{}\n".format( words.shape[0]) )
	fout.close()

	if scores == None:
		# scores for inv File
		scores = qe.scores_for_invertedFile( words )

	# build retrieved list of images
	return ranker.build_retrieved_list(scores)



def get_all_rankings( ranker, assignments_db, query_extractor, list_queries, settings, path_workspace=None, path_store_list=None):

	res = []
	foutq = open( "queries.txt", "w+" )
	fout = open( "map.txt", "w+" )
	for k, frame in enumerate(list_queries):
		frame = os.path.basename(  frame  )
		fr =  frame.split('.jpg')[0]
		topic = query_extractor.queries.parse_topic_fr[ fr ]

		final, final_scores = get_ranking( ranker, query_extractor, os.path.join( settings["path_queries"], frame) )

		if settings[ "dataset" ] == 'trecvid_subset':
			 final, shots, final_scores = build_retrieved_list_trecvid( final, final_scores)
			 f_id = os.path.basename( frame ).split('.')[1]
			 # save lists
			 store_trecvid_rankings( topic+'_'+f_id, final, shots, final_scores, path_out=path_store_list )
			 ap = AveragePrecision( "/media/eva/Eva Data/Datasets/ins15/8_groundtruth/src/ins.search.qrels.tv13", int(topic), shots )

			 print "{}\t{}".format( fr, ap )

		else:
			store_list( topic, final, path_out=path_store_list )
			ap = evaluate_one_frame( topic, path_workspace, settings, path_lists=path_store_list )
			print "{}\t{}".format( topic, ap )

		foutq.write("{}\n".format(topic))
		fout.write("{}\n".format(ap))
		res.append( ap )

	foutq.close()
	fout.close()
	return res


def main():
	# for sumpooling
	#dataset_name = "paris"
	#dataset_name = "paris106k"
	#dataset_name = "oxford"
	#dataset_name = "oxford105k"
	dataset_name = "trecvid_subset"


	network="vgg16"
	list_layers=['conv5_1']
	path_workspace = "/home/eva/Workspace/icmr_pipeline/"
	#settings=load_settings("settings.json")
	settings=load_settings(os.path.join( path_workspace, "{}/settings.json".format(dataset_name)) )

	#dim_input="336_256" #our
	#dim_input="586_586" # babenko
	#dim_input="672_512" # double ours
	#dim_input="1024_768" # yannis
	dim_input="384_288" #trecvid
	#dim_input="768_576"
	# check this!!

	#new_dim=(21,16) # ours simple
	#new_dim=(42,32) # ours bow
	#new_dim=(32,24)
	#new_dim=(37,37) # babenko
	new_dim=(48,36) # trecvid bow
	#new_dim = (24,18) #trecvid pooling
	masking = 3
	augmentation = [0]

	path_store_list = os.path.join( path_workspace, dataset_name,"lists_bow/{}/{}/{}".format(  network, dim_input, list_layers[0] ) )

	path_queries=settings["path_queries"]
	path_output=str(settings["path_output"])
	path_models=str(settings["path_models"])

	pathDB=os.path.join(path_output,'features',network, dim_input)

	pca_model_name =os.path.join(path_models, 'pca',network, dim_input, 'vgg16_conv5_1_25000_512_pca.pkl')
	#pca_model_name=None
	model_centroids=os.path.join(path_models,'centroids', network, dim_input,'vgg16_conv5_1_25000_512_pca.pkl')


	#assignments_db=os.path.join(path_output,'assignments', network, dim_input, 'vgg16_conv5_1_25000_42_32_Weight_512_pca_db')
	assignments_db=os.path.join(path_output,'assignments', network, dim_input, 'vgg16_conv5_1_25000_48_36_NoWeight_512_pca_db')
	path_index=os.path.join(path_output,'invertedIndex', network, dim_input)

	input_size=(1,3, int(dim_input.split('_')[0]),  int(dim_input.split('_')[1]) )

	print
	print "RANKING {}".format( settings[ "dataset" ] )
	print "----------------------"
	print
	print "Reading from: {}".format(pathDB)
	print "Layers: {}".format(list_layers)
	print "PCA: {}".format(pca_model_name)
	print "Centroids: {}".format(model_centroids)
	print "Assignments: {}".format(assignments_db)
	print "Inverted file: {}".format(path_index)
	print "Input network: {}".format(input_size)
	print "Masking: {}".format(masking)
	print
	s = raw_input("--> Press key")




	# rm prev results
	clean_folder(  path_store_list )
	invfile_path= os.path.join( path_index, assignments_db.split('/')[-1].split('_db')[0]+'.dat' )
	dic_inv= os.path.join( path_index, assignments_db.split('/')[-1].split('_db')[0]+'.npy' )

	ranker = Ranker( invfile_path,  dic_inv  )

	# init object to read and preocess features from leveldb
	query_extractor = Query_Extractor(model_centroids, settings, network, list_layers, input_size, new_dim=new_dim, masking=masking, augmentation=augmentation, pca_model_name=pca_model_name )

	# list Queries_paris
	keyframes_q, labels_q = create_list_queries(settings)

	res = get_all_rankings( ranker, assignments_db, query_extractor, keyframes_q, settings, path_workspace, path_store_list=path_store_list)
	print np.average( res )


if __name__=="__main__":
	main()

	"""
	if settings[ "dataset" ] == "trecvid_subset":
		data = np.loadtxt( open(os.path.join( path_store_list, "results.txt" ), 'r' ), dtype='str' )[:,3].astype(np.float32)
		print "Final mAP"
		print np.average( data )

	else:
		evaluate_results( path_workspace, settings, path_lists=path_store_list )
	"""
