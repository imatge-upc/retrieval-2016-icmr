import os, glob, sys
import time
import logging
import numpy as np
import caffe
import json
import pickle
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.measurements import center_of_mass
import math
import cv2

def get_distanceTransform(mask):
    img = (255*mask).astype(np.uint8)
    dist = cv2.distanceTransform(255-img, cv2.cv.CV_DIST_L2, cv2.cv.CV_DIST_MASK_PRECISE)
    #mask_weights = np.exp(-0.05*dist)
    dist = 1 - dist/np.max(dist)
    return dist

def check_verHor( image, dim ):
    # check vertical/horitz
    if image.shape[0]>image.shape[1] and dim[0]<dim[1]:
        dim = ( dim[1], dim[0] )

    elif image.shape[0]<image.shape[1] and dim[0]>dim[1]:
        dim = ( dim[1], dim[0] )
    return dim

def clean_folder( folder ):
    # rm prev results
    files = glob.glob( os.path.join( folder, "*" ) )
    print "Cleaning {} files from folder {}".format( len(files), folder )
    s = raw_input("--> Press key")

    if len(files) > 0:
        for f in files:
            os.remove(f)
    if not os.path.exists(folder):
        os.makedirs( folder )


def gaussian_weights(shape, center = None, sigma=None ):
    r1 = shape[0] //2
    r2 = shape[1] //2


    ys = np.linspace(-r1, r1, shape[0])
    xs = np.linspace(-r2, r2, shape[1])
    YS, XS = np.meshgrid(xs, ys)

    if center is not None:
        YS -= ( center[1]-r1 )
        XS -= (center[0]-r2 )

    if sigma is None:
        sigma = min(shape[0], shape[1]) / 3.0
    g = np.exp(-0.5 * (XS**2 + YS**2) / (sigma**2))
    return g


def get_weighted_map( mask, mode='gaussian', sigma=None ):
    """
    Get weighted distance map to the center point of the foreground object
    """
    c =  np.array(center_of_mass(mask)).astype(np.int)

    if mode == 'gaussian':
        dist_m = gaussian_weights( mask.shape, center=c, sigma=sigma )
    elif mode == 'distance':
        dist_m = weighted_distances( dx=mask.shape[0], dy=mask.shape[1], c=(c[0],c[1]) )
    dist_m[ mask==1 ] = 1
    return dist_m

def get_bbx( mask ):
    # get coordinates
    x_c, y_c = np.where(mask==1)

    x_min = np.min(x_c)
    x_max = np.max(x_c)

    y_min = np.min(y_c)
    y_max = np.max(y_c)

    h = x_max-x_min
    w = y_max-y_min

    return x_min, y_min, h, w


def conver_bbx_mask( mask, dim ):
    """
    take original mask, and desired dimensions (order dim sorted already!)
    and converts it into a mask with fg object within a bbx
    """

    # Find coordinates
    coord = np.where(mask ==1)

    # Find values
    ymin = np.min(coord[0])
    xmin = np.min(coord[1])
    ymax = np.max(coord[0])
    xmax = np.max(coord[1])

    # Find multipliers to convert to assignment map size
    multi_x = float(dim[1])/np.shape(mask)[1]
    multi_y = float(dim[0])/np.shape(mask)[0]

    # Define query box as [xmin,ymin,xmax,ymax]
    query_box = [int(math.floor(xmin*multi_x)),int(math.floor(ymin*multi_y)),int(math.ceil(xmax*multi_x)),int(math.ceil(ymax*multi_y))]

    return query_box



def increase_image( image, percentage=0.50, debug=False ):

    # get original dim
    dim_original = np.array(image.shape[:2])

    # new dimensions
    dim_new = np.around(dim_original*( 1.0+percentage ))
    if debug:
        print "original: {}".format( dim_original )
        print "new: {}".format( dim_new )

    # reshape image
    if len(image.shape) == 3:
        new_image = caffe.io.resize_image( image, dim_new )

    elif len(image.shape) == 2:
        new_image = caffe.io.resize( image, dim_new )

        # make sure is binary mask
        new_image[ new_image > 0.5 ] = 1.0
        new_image[ new_image <= 0.5 ] = 0.0

    else:
        print "Error in the image/mask dimensions!"

    return new_image

def get_center_crop( image, percentage=0.50, debug=False ):

    # get original dim
    dim_original = np.array(image.shape[:2])

    # new dimensions
    dim_new = np.around(dim_original*( 1.0+percentage ))
    if debug:
        print "original: {}".format( dim_original )
        print "new: {}".format( dim_new )

    # reshape image
    if len(image.shape) == 3:
        new_image = caffe.io.resize_image( image, dim_new )

    elif len(image.shape) == 2:
        new_image = caffe.io.resize( image, dim_new )

        # make sure is binary mask
        new_image[ new_image > 0.5 ] = 1.0
        new_image[ new_image <= 0.5 ] = 0.0

    else:
        print "Error in the image/mask dimensions!"

    # center
    c = dim_new / 2.0
    DX = np.around( dim_original/2.0 )
    # crop
    crop = new_image[ c[0]- DX[0]:c[0]+DX[0] , c[1] - DX[1]: c[1] + DX[1] ]

    return crop


def weighted_distances( dx=10, dy=10, c=(5,5)):
    '''
    Map with weighted distances to a point
	args: Dimension maps and point
    '''

    a = np.zeros((dx,dy))
    a[c]=1

    indr = np.indices(a.shape)[0,:]
    indc = np.indices(a.shape)[1,:]

    difr = indr-c[0]
    difc = indc-c[1]

    map_diff = np.sqrt((difr**2)+(difc**2))

    map_diff = 1.0 - (map_diff/ map_diff.flatten().max())

    # Return inverse distance map
    return map_diff

def weight_assignments( assignments, weights, K, full_vector=False):
    """
    Return tupe with (ass, weight)
	args: Assignments, weigts (both flatten np.array) and
	      K as number of centroids to generate histograms
    """
    ids_words = np.unique(assignments)
    map_score = []
    for id_w in ids_words:
        map_score.append( np.sum( weights[assignments==id_w] ) )
    map_score = np.array(map_score)

    res = np.zeros(K)
    for k, vw in enumerate(ids_words):
            res[vw]= map_score[k]

    # l2-normalize vector
    res = res/np.sqrt(sum(res**2))
    idx = np.where( res>0)[0]

    # in case we want the full histogram
    if full_vector:
        return res

    # stuff for the inverted file
    else:
        return zip(idx,res[idx])

def loadTFIDF(assignments_db,path_tfidf):
    main_path = path_tfidf
    name_model = assignments_db.split('/')[-1].split('_db')[0]+'_tfidf.npy'

    if not os.path.exists( os.path.join( main_path, name_model) ):
        return None
    else:
        # save indices to map id (from
        return load_obj( os.path.join( main_path, name_model) )

def flipImage( ima, horitzontal=True, vertical=False ):
    if len(ima.shape) == 3:
        if horitzontal and vertical:
            return ima[::-1,::-1,:]
        elif horitzontal:
            return ima[:,::-1,:]
        elif vertical:
            return ima[::-1,:,:]
    else:
        if horitzontal and vertical:
            return ima[::-1,::-1]
        elif horitzontal:
            return ima[:,::-1]
        elif vertical:
            return ima[::-1,:]

def check_path_file(file, create_if_missing=True):
    """
    Check that if the file path exits and if not create it
    """
    path_file = os.path.dirname(file)
    if not os.path.exists(path_file):
        os.makedirs( path_file )
        return False
    else:
        return True

def get_bow_hist(assignments, K):
    """
    Create l2-norm histogram from assignemnts.
        assignemnts: arra
    """
    # init vector to the total number of clusters
    res = np.zeros(K)

    # Count words
    for a in assignments:
        res[a]+=1

    # l2-normalize vector
    res = res/np.sqrt(sum(res**2))
    return res

def get_histogram_counts( assignments, K ):
    res =  get_bow_hist(assignments, K)
    # get index only forpositions > 0 (forinv.index storage)
    idx = np.where( res>0)[0]
    return zip(idx,res[idx])

def get_dimensions( net, list_layers):
    dim_layer = {}
    for label in list_layers:
       dim_layer[label] = net.blobs[label].data.shape[1:]
    del net
    return dim_layer

# For pickle objects  -- storing indices (python dict)
def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open( name , 'rb') as f:
        return pickle.load(f)

def reshape_maps_zoom( maps, new_dim, interp_order=1 ):
    res = []
    for k in range(maps.shape[0]):
        f_map = maps[k,:,:]
        scale = tuple(np.array(new_dim, dtype=float))/np.array(f_map.shape)
        out = zoom( f_map, scale, order=interp_order )
        res.append(  out  )
    return np.array(res)

# Reshape feature maps
def reshape_maps( maps, new_dim, interp_order=1 ):
    res = []
    for k in range(maps.shape[0]):
        f_map = maps[k,:,:]
        f_map = np.expand_dims(f_map, axis=2)
        res.append(caffe.io.resize_image(f_map, new_dim ).squeeze())
    return np.array(res)

# Read keyframes with label for aspect ratio
def get_all_keyframes( settings ):

    list_vertical=str(settings["keyframesVertical"])
    list_horitzontal=str(settings["keyframesHoritzontal"])

    all_keyframes = []

    k_v = load_paths_txt( list_vertical )
    k_h = load_paths_txt( list_horitzontal )

    all_keyframes.extend(k_v)
    all_keyframes.extend(k_h)

    labels = np.ones((len(k_v)+len(k_h)))
    labels[len(k_v):]=0
    return all_keyframes, labels==1


def load_batch_images( list_batch ):
    ima_batch = []
    for name in list_batch:
        ima_batch.append(caffe.io.load_image( name ))
    return ima_batch

def load_settings( filename ):
    with open( filename, 'r' ) as f:
        data = json.load(f)
    return data

def set_logger( name_logger, level_logger="info", format_logger="%(asctime)s -%(levelname)s - %(message)s"):
    log = logging.getLogger(name_logger)
    if level_logger == "info":
        level = logging.INFO
    else:
        level = logging.DEBUG

    log.setLevel(level)
    #create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)
    #create formatter
    #formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    formatter = logging.Formatter( format_logger )
    #add formatter to ch
    ch.setFormatter(formatter)
    #add ch to logger
    log.addHandler(ch)

    return log

def load_paths_txt( filename ):
    path_list = []
    with open(filename, 'r') as fid:
        for line in fid:
            path_list.append( line.split('\n')[0] )
    return np.array(path_list)

class ProcessingStatus():
    def __init__(self, batch_size=1.0, n_total=None):
        self.time_ref = 0.0
        self.counter = 0
        self.batch_size = batch_size
        self.n_total = n_total

    def start(self):
        self.time_ref = time.time()
        message = "{} images to process".format(self.n_total)
        return message

    def update(self, extra_info=''):
        self.counter+=self.batch_size
        t = time.time()-self.time_ref
        speed_image = float(t)/self.batch_size
        message = "{} images, \t {:.2f} seg/image, \t{}\n".format(self.counter, float(speed_image), extra_info)
        self.time_ref = time.time()
        return message

    def lastBatch(self, n_images, extra_info=''):
        self.counter+=n_images
        t = time.time()-self.time_ref
        speed_image = float(t)/float(n_images)
        message = "[LAST BATCH] {} images, \t{:.2f} seg/image, \t{} \n".format( self.counter, float(speed_image), extra_info )
        return message

    def get_count(self):
        return self.counter
