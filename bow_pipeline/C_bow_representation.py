from reader import *
from utils import *
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import skcuda.linalg as culinalg
import skcuda.misc as cumisc
import numpy as np
from invidx import InvertedIndex, InvertedIndexBuilder
import time
import pickle
from collections import Counter
from gensim import corpora, models, similarities



class GPUCodebook(object):
    """
    GPU version of the codebook. Much faster on the Titan X: around 15000
    512D codewords per sec
    """
    def __init__(self, centers):
        culinalg.init()

        self.centers = centers.astype(np.float32)
        self.centers_gpu = gpuarray.to_gpu(self.centers)
        self.center_norms = cumisc.sum(self.centers_gpu**2, axis=1)

    def get_distances_to_centers(self, data):

        # make sure the array is c order
        data = np.asarray(data, dtype=np.float32, order='C')

        # ship to gpu
        data_gpu = gpuarray.to_gpu(data)

        # alloc space on gpu for distances
        dists_shape = (data.shape[0], self.centers.shape[0])
        dists_gpu = gpuarray.zeros(dists_shape, np.float32)

        # calc data norms on gpu
        data_norms = cumisc.sum(data_gpu**2, axis=1)

        # calc distance on gpu
        cumisc.add_matvec(dists_gpu, self.center_norms, 1, dists_gpu)
        cumisc.add_matvec(dists_gpu, data_norms, 0, dists_gpu)
        culinalg.add_dot(data_gpu, self.centers_gpu,
            dists_gpu, transb='T', alpha=-2.0)
        return dists_gpu

    def get_assignments(self, data):
        dists = self.get_distances_to_centers(data)
        return cumisc.argmin(dists, 1).get().astype(np.int32)

    @property
    def dimension(self):
        return self.centers.shape[1]


# from 'util_oxford' package
def saveIndex(assignments_db,path_index, index, indices, tfidf=False):
    main_path = path_index

    if not tfidf:
    	name_invFile = assignments_db.split('/')[-1].split('_db')[0]+'.dat'
    	name_indices = assignments_db.split('/')[-1].split('_db')[0]+'.npy'
    else:
     	name_invFile = assignments_db.split('/')[-1].split('_db')[0]+'tfidf.dat'
    	name_indices = assignments_db.split('/')[-1].split('_db')[0]+'tfidf.npy'

    if not os.path.exists(main_path):
        os.makedirs(main_path)
    # save index
    index.save( os.path.join( main_path, name_invFile) )

    # save indices to map id (from
    save_obj( indices, os.path.join( main_path, name_indices) )
    return os.path.join(main_path, name_invFile), os.path.join(main_path, name_indices)

def saveTFIDF(assignments_db,path_tfidf, model):
    main_path = path_tfidf
    name_model = assignments_db.split('/')[-1].split('_db')[0]+'_tfidf.npy'

    if not os.path.exists(main_path):
        os.makedirs(main_path)

    # save indices to map id (from
    save_obj( model, os.path.join( main_path, name_model) )



log = set_logger("C_bow_representations", level_logger="info", format_logger="%(asctime)s- %(name)s - %(levelname)s - %(message)s")

###################################################################
# Compute assignments and generate inferted file
###################################################################

if __name__=="__main__":

    '''
    Paths
    '''
    # oxford
    settings=load_settings("/home/eva/Workspace/icmr_pipeline/oxford/settings.json")

    # oxford105K
    #settings=load_settings("/home/eva/Workspace/icmr_pipeline/oxford105k/settings.json")


    # paris
    #settings = load_settings("/home/eva/Workspace/icmr_pipeline/paris/settings.json")

    # paris
    #settings = load_settings("/home/eva/Workspace/icmr_pipeline/paris106k/settings.json")

    # trecvid_subset
    #settings = load_settings("/home/eva/Workspace/icmr_pipeline/trecvid_subset/settings.json")

    dim_input="336_256"
    #dim_input="586_586"

    #dim_input="384_288" trecvid
    #dim_input="672_512"

    network="vgg16"
    path_output=str(settings["path_output"])
    path_models=str(settings["path_models"])

    pathDB=os.path.join(path_output,'features',network, dim_input)
    list_layers=['conv5_1']

    pca_model_name =os.path.join(path_models, 'pca',network, dim_input, 'vgg16_conv5_1_25000_512_pca.pkl')
    #pca_model_name=None
    model_centroids=os.path.join(path_models,'centroids', network, dim_input,'vgg16_conv5_1_25000_512_pca.pkl')

    assignments_db=os.path.join(path_output,'assignments', network, dim_input, 'vgg16_conv5_1_25000_42_32_NO_Weight_512_pca_db')
    path_index=os.path.join(path_output,'invertedIndex', network, dim_input)

    keyframes, labels = get_all_keyframes( settings )




    print
    print "Reading from: {}".format(pathDB)
    print "Layers: {}".format(list_layers)
    print "PCA: {}".format(pca_model_name)
    print "Centroids: {}".format(model_centroids)
    print "Assignments: {}".format(assignments_db)
    print "Inverted file: {}".format(path_index)
    print "Number images: {}".format(len(keyframes))
    print
    s = raw_input("--> Press key")


    #new_dim=(48,36) #trecvid
    new_dim=(42,32)
    #new_dim=(37*2,37*2)

    #new_dim=(21,16)
    weighted_maps=False # Change that for trecvid
    tfidf=False
    path_tfidf=os.path.join(path_output,'tfidf', network, dim_input)


    status = ProcessingStatus()

    centers = np.load( model_centroids )
    log.info( "Lodaded centers {}".format(centers.shape) )

    index = InvertedIndexBuilder(centers.shape[0])
    indices = {}
    log.info( "Init Inverted File".format(centers.shape) )
    codebook = GPUCodebook(centers)
    log.info( "Init GPU Codebook object".format(centers.shape) )

    # dimensions codebook
    K = centers.shape[0]
    t0 = time.time()

    # init reader
    reader = Local_Feature_ReaderDB( pathDB, list_layers, applyPCA=pca_model_name)


    # ensure that we only take filename without path
    keyframes = map( os.path.basename, keyframes )

    # build dictionary with labels per keyframe
    map_labels = {}
    for keyframe, label in zip(keyframes, labels):
        map_labels[keyframe]=label


    count = 0

    flag_assignments=False
    if not os.path.exists( assignments_db ):
        # Generate assignments
        os.makedirs( assignments_db )
        log.info("Computing assignments from scratch!")
        flag_assignments=True

        # Init levelDB
        db = leveldb.LevelDB( assignments_db )
        keyframes_done=[]
    else:
        # Init levelDB
        db = leveldb.LevelDB( assignments_db )

    # Check the keyframes already computed
    keyframes_done = set([key for key in db.RangeIter(include_value=False) ])
    vals = [np.fromstring(value, dtype=np.int32) for key, value in db.RangeIter() ]
    log.info("Assignments db existing.")

    # Check images still to compute assignments...
    if len(keyframes_done)==len(keyframes):
        flag_assignments=False
        log.info("Assignments completed!")
    else:
        flag_assignments=True
        keyframes=set(keyframes)-keyframes_done
        log.info("Finishing assingments. {} images to compute.".format(len(keyframes)))



    # ASSIGNMENTS COMPUTATION --------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------
    if flag_assignments:
        count=0
        if len(keyframes_done)>0:
            count=len(keyframes_done)

        t0=time.time()
        #loop over all the images...
        for i, key in enumerate(keyframes):
            # read features - Setting new dimension if requiered
            feats = reader.get_features(key, vertical=map_labels[key], new_dim=new_dim )
            assignments = codebook.get_assignments(feats) # int32
            # store assignment
            db.Put(key, assignments.tostring())

            # update count
            count+=1

            # display status
            if count%100==0:
                log.info("processing {} of {}\t total done {} ".format(i+1, len(keyframes), count))
        t1=time.time()
        log.info("Assignments elapsed time={}s\n\n".format(t1-t0))


    # BUILDING INVERTED FILE ---------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------
    # init counter
    count=0

    log.info("Generating Inverted file...")
    for key, value in db.RangeIter():
        t0=time.time()
        assignments = np.fromstring(value, dtype=np.int32)
        log.debug("checking size {}".format(assignments.shape))
        if weighted_maps:
            log.debug("weighted...")
            # create distance maps
            if map_labels[key]:
                maps = weighted_distances( new_dim[0],new_dim[1], c=(new_dim[0]/2,new_dim[1]/2) )
            else:
                maps = weighted_distances( new_dim[1],new_dim[0], c=(new_dim[1]/2,new_dim[0]/2) )
                # get scores for assignments
            scores = weight_assignments( assignments, maps.flatten(), K )
        else:
            # l2-normalize word frequency to generate scores
            scores = get_histogram_counts( assignments, K )

        # index scores..
        index.insert( scores )
        indices[count]=os.path.basename(key)
        count+=1
        # display status
        if count%100==0:
            #log.info(status.update())
            log.info( "{} images of {} processed".format( count, len(keyframes) ) )
            log.debug( "\tword count {}".format( index.word_count ) )
            log.debug( "\tdocument count {}".format( index.document_count ))
            log.debug( "\tentry count {}".format( index.entry_count ) )

        status.update()
        t1=time.time()

    # save the index
    path_inv, path_indices = saveIndex( assignments_db,path_index, index, indices )
    log.info( 'Inverded index completed!\n\tElapsed time {}s'.format(t1-t0) )
    log.info("{}\n{}".format(path_inv, path_indices))

    index=InvertedIndex(path_inv)
    print index.lookup(18593)


    # if TFIDF ...
    if tfidf:
        if loadTFIDF(assignments_db,path_tfidf) is None:
            print "Computing tfidf..."
            corpus = []

            for k, v in db.RangeIter():
                counter = Counter( np.fromstring(v, dtype=np.int32) )
                corpus.append( zip(counter.elements(), counter.values()) )

            tfidf_model = models.TfidfModel(corpus)
            saveTFIDF(assignments_db,path_tfidf, tfidf_model)
            print 'Done! '
        else:
            print "loading TFIDF weights ... "
            index = InvertedIndexBuilder(centers.shape[0])
            indices = {}
            log = set_logger("C_bow_representations", level_logger="info", format_logger="%(asctime)s- %(name)s - %(levelname)s - %(message)s")
            status = ProcessingStatus()
            tfidf_model=loadTFIDF(assignments_db,path_tfidf)

            count=0
            #loop over all the image assignments...
            for key, v in db.RangeIter():
                print key
                counter = Counter( np.fromstring(v, dtype=np.int32) )
                # already l2-normalized ?
                scores_tfidf = tfidf_model[ zip(counter.elements(), counter.values()) ]

                '''
                # normalize scores anyway
                hist = np.zeros(K)
                for word, score in scores_tfidf:
                    hist[word]=score

                idx = np.where( hist>0)[0]
                scores = zip(idx,hist[idx])
                '''
                index.insert( scores_tfidf )
                indices[count]=os.path.basename(key)
                count+=1

            # save the index
            saveIndex( assignments_db,path_index, index, indices, tfidf=True )
            log.info( 'Inverded index completed!' )
