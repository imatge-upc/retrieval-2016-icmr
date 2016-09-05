import glob, sys, os
import numpy as np
import collections
from scipy.ndimage import imread
from scipy.misc import imsave
import leveldb
import caffe
sys.path.insert(0, '/home/eva/Workspace/icmr_pipeline/bow_pipeline')
from utils import *
import cv2



###########################################################
# This class is specific to the dataset, it gets topics and
# relevant images based on data provided in the webpage
###########################################################

class Queries_paris():
    def __init__(self, path_gt, path_images):

        # path_gt
        self.path_gt = path_gt
        #path images
        self.path_images = path_images
        # dictionary to parse query name with image path
        self.indx = collections.defaultdict(list)
        # dictionary to parse query name with bounding box coordinates
        self.bbx =  collections.defaultdict(list)
        # dictionary to parse query name with list of their relevant image filenames
        self.relevant_images = collections.defaultdict(list)

        # dictionary to parse name topic frame to ID topic frame (for eval)
        self.parse_topic_fr = {}
        # fill dict for the queries (5 images per topic)

        for f_name in glob.glob( os.path.join(path_gt, "*query.txt") ):
            file_parts = os.path.basename(f_name).split('_')

            q_id = file_parts[0]+"_"+file_parts[1]

            # read only one line per file
            try:
                data = np.loadtxt(f_name, dtype="str")
                data.shape
            except:
                print "Query empty! {}".format( f_name )

            self.indx[q_id].append( os.path.join( data[0] ) )
            self.parse_topic_fr[ data[0] ] = os.path.basename(f_name).split('_query')[0]
            self.bbx[q_id].append( data[1:].astype(float).astype(int) )

        # fill dict for relevant images per topic
        for label in ["*good.txt", "*ok.txt", "*junk.txt"]:
            # Here we take all queries for a label and iterate accross them...
            for f_name in glob.glob( os.path.join( path_gt, label ) ):
                # this is for taking the name of the query
                file_parts = os.path.basename( f_name ).split('_')

                q_id = file_parts[0]+"_"+file_parts[1]
                # here we load all images in the txt file: [IMPORTANT] - Make ndmin ==1, otherwise problems when file only have one row.
                if not os.stat( f_name ).st_size == 0:
                    data = np.loadtxt( f_name, dtype='str', ndmin=1 )
                    for relevant in data:
                        if relevant not in np.array(self.relevant_images[ q_id ]):
                            try:
                                self.relevant_images[ q_id ].extend( data )
                            except:
                                self.relevant_images[ q_id ].append( data )
                else:
                    print "empty! {} {}".format( f_name, label )

    def get_mask_frame( self, f_name_image, dim=None):
    	""" Get mask for a given query and resize to the feature map dimension
    	desired """
        # frame ID without path and extension
        frame = os.path.basename( f_name_image ).split('.')[0]
        ima = caffe.io.load_image( f_name_image )

        # get real image filename for the query
        f_name=os.path.join(self.path_gt, self.parse_topic_fr[frame]+'_query.txt')

        # get bbx coordinates
        bbx = np.loadtxt(f_name, dtype="str")[1:].astype(float).astype(int)

        # check vertical/horitz
        if dim is not None:
            if ima.shape[0]>ima.shape[1]:
                dim_ = dim
            else:
                dim_ = (dim[1], dim[0])
        else:
            dim_ = ima.shape[:2]

        # init output mask
        mask = np.zeros( ima.shape[:2] )
        mask[  bbx[1]:bbx[3], bbx[0]:bbx[2]]=1
        #print "debug mask ", mask.shape

        # resize output to specified dimensions
        if dim !=None:
            mask_r = reshape_maps( np.expand_dims(mask, axis=0 ) , dim_).squeeze()
        else:
            mask_r = mask
    	#print "debug mask2 ", mask_r.shape, np.unique( mask )

        return mask_r

    def get_topics(self):
        return self.indx.keys()

    def get_frames(self, q_id):
        return self.indx[q_id]

    def get_bbx(self, q_id):
        return self.bbx[q_id]

    def get_relevant_images_ids(self, q_id):
        return self.relevant_images[q_id]
    @classmethod
    def from_settings(cls, settings):
        return cls(str(settings["path_gt_files"]), str(settings["path_images"]))


class Queries_oxford():
    def __init__(self, path_gt, path_images):
        # path_gt
        self.path_gt = path_gt
        #path images
        self.path_images = path_images
        # dictionary to parse query name with image path
        self.indx = collections.defaultdict(list)
        # dictionary to parse query name with bounding box coordinates
        self.bbx =  collections.defaultdict(list)
        # dictionary to parse query name with list of their relevant image filenames
        self.relevant_images = collections.defaultdict(list)

        # dictionary to parse name topic frame to ID topic frame (for eval)
        self.parse_topic_fr = {}
        # fill dict for the queries (5 images per topic)

        for f_name in glob.glob( os.path.join(path_gt, "*query.txt") ):
            file_parts = os.path.basename(f_name).split('_')
            # in case the query ID contains two words...
            if len(file_parts)==4:
                q_id = file_parts[0]+"_"+file_parts[1]
            else:
                q_id = file_parts[0]

            # read only one line per file
            data = np.loadtxt(f_name, dtype="str")

            self.indx[q_id].append( os.path.join( data[0].split('oxc1_')[1]) )
            self.parse_topic_fr[ os.path.join( data[0].split('oxc1_')[1]) ] = os.path.basename(f_name).split('_query')[0]
            self.bbx[q_id].append( data[1:].astype(float).astype(int) )

        # fill dict for relevant images per topic
        for label in ["*good.txt", "*ok.txt", "*junk.txt"]:
            # Here we take all queries for a label and iterate accross them...
            for f_name in glob.glob( os.path.join( path_gt, label ) ):
                # this is for taking the name of the query
                file_parts = os.path.basename( f_name ).split('_')

                # if the query has two words or one...
                if len( file_parts )==4:
                    q_id = file_parts[0]+"_"+file_parts[1]
                else:
                    q_id = file_parts[0]

                # here we load all images in the txt file: [IMPORTANT] - Make ndmin ==1, otherwise problems when file only have one row.
                data = np.loadtxt( f_name, dtype='str', ndmin=1 )

                for relevant in data:
                    if relevant not in np.array(self.relevant_images[ q_id ]):
                        try:
                            self.relevant_images[ q_id ].extend( data )
                        except:
                            self.relevant_images[ q_id ].append( data )

    def get_mask_frame( self, f_name_image, dim=None):
    	""" Get mask for a given query and resize to the feature map dimension
    	desired """
        # frame ID without path and extension
        frame = os.path.basename( f_name_image ).split('.')[0]
        ima = caffe.io.load_image( f_name_image )

    	# get real image filename for the query
    	f_name=os.path.join(self.path_gt, self.parse_topic_fr[frame]+'_query.txt')

    	# get bbx coordinates
    	bbx = np.loadtxt(f_name, dtype="str")[1:].astype(float).astype(int)

    	# check vertical/horitz
    	if dim is not None:
    		if ima.shape[0]>ima.shape[1]:
    			dim_ = dim
    		else:
    			dim_ = (dim[1], dim[0])
    	else:
    		dim_ = ima.shape[:2]

    	# init output mask
    	mask = np.zeros( ima.shape[:2] )
    	mask[  bbx[1]:bbx[3], bbx[0]:bbx[2]]=1
    	#print "debug mask ", mask.shape

    	# resize output to specified dimensions
    	if dim !=None:
    		mask_r = reshape_maps( np.expand_dims(mask, axis=0 ) , dim_).squeeze()
    	else:
    		mask_r = mask
    	#print "debug mask2 ", mask_r.shape, np.unique( mask )

    	return mask_r

    def get_topics(self):
        return self.indx.keys()

    def get_frames(self, q_id):
        return self.indx[q_id]

    def get_bbx(self, q_id):
        return self.bbx[q_id]

    def get_relevant_images_ids(self, q_id):
        return self.relevant_images[q_id]
    @classmethod
    def from_settings(cls, settings):
        return cls(str(settings["path_gt_files"]), str(settings["path_images"]))



class Queries_trecvid(object):
    def __init__(self, abs_path=None):
        if abs_path==None:
            self.abs_path = "/media/eva/Eva Data/Datasets/ins15/topics/2013/tv13.example.images"
        else:
            self.abs_path = abs_path

        self.parse_topic_fr = {}

        # build parser from filename to topic
        for file in glob.glob( os.path.join( self.abs_path, '*.png') ):
            topic = os.path.basename(file).split('.')[0]
            self.parse_topic_fr[ os.path.basename(file) ]=topic

    def get_src_path_fromID(self, ID, frameID):
        return self.abs_path+'/'+str(ID)+'.'+str(frameID )+'.src.png'

    def get_mask_path_fromID(self, ID, frameID):
        return self.abs_path+'/'+str(ID)+'.'+str(frameID )+'.mask.png'


    def get_mask_frame( self, frame, dim=None  ):
        '''
        return mask and name of the query frame. OUTDATED FUNCTION
        '''
        filename = frame.replace(".src.", ".mask.")

        # read image
        ima = cv2.imread(filename)

        # make sure is a mask
        if len(ima.shape)>2:
            ima = ima[:,:,0]

        # binarise
        ima[ima >0]=1.0

        # check dims
        if dim is not None:
            if ima.shape[0]>ima.shape[1]:
                dim_ = dim
            else:
                dim_ = (dim[1], dim[0])
        else:
            dim_ = ima.shape[:2]

        mask_r = reshape_maps_zoom( np.expand_dims(ima, axis=0 ) , dim_).squeeze()
        mask_r[mask_r >0]=1.0

        return mask_r

    def get_image_frame( queries, topic, id_frame ):
        return imread( self.get_src_path_fromID(topic, id_frame) )




###########################################################
# Additional functions
###########################################################


def create_list_queries( settings ):
    if settings[ "dataset" ] == "trecvid_subset":
        return create_list_queries_trecvid( settings )
    else:
        return create_list_queries_oxf_par( settings )

def create_list_queries_oxf_par( settings ):
    """ Generates list with query names and their horitz/vert labels
        abs_path to the query and label"""

    path_gt = settings["path_gt_files"]
    path_images = settings["path_images"]
    dataset = settings["dataset"]

    keylist=[]
    labels=[]

    # make abs path to query images
    list_queries = glob.glob( os.path.join(path_gt, '*_query*') )

    # for each image, check ressolution...
    for file in list_queries:
        if dataset == "oxford":
            name = np.loadtxt( file, dtype='str' )[0].split('oxc1_')[1]
        elif dataset == "paris":
            name = np.loadtxt( file, dtype='str' )[0]

        ima=caffe.io.load_image(os.path.join(path_images, name+'.jpg'))
        keylist.append( os.path.join(path_images, name+'.jpg' ) )

        if ima.shape[0]>ima.shape[1]:
            labels.append(True)
        else:
            labels.append(False)

    # return keynames and labels
    return np.array(keylist), np.array(labels)


def create_list_queries_trecvid( settings ):
    """ Generates list with query names and their horitz/vert labes
        abs path to the query and label"""

    topics=[]
    frames = []
    labels=[]
    # make abs path to query images
    list_queries = []

    a = np.loadtxt( settings[ "path_gt_file" ], dtype=str )
    a =  np.unique(a[:,0].astype(int))

    for i in a:
        for j in range(1,5):
            list_queries.append( os.path.join( settings[ "path_queries" ], str(i)+'.'+str(j)+'.src.png' ) )

    # for each image, check ressolution...
    for query in list_queries:
        ima=caffe.io.load_image( query )
        if ima.shape[0]>ima.shape[1]:
            l = True
        else:
            l = False

        labels.append(l)


    # return keynames and labels
    return np.array(list_queries), np.array(labels)
