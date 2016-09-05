import numpy as np
import leveldb
import json
import os




#path_db = "/media/eva/Eva Data/icmr_results/oxford105k/features/vgg16/1024_768/pool5_db"
#path_settings = "/home/eva/Workspace/icmr_pipeline/oxford105k/settings.json"


def load_settings( filename ):
    with open( filename, 'r' ) as f:
        data = json.load(f)
    return data

def load_paths_txt( filename ):
    path_list = []
    with open(filename, 'r') as fid:
        for line in fid:
            path_list.append( line.split('\n')[0] )
    return np.array(path_list)

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

#keyframes_done = set([key for key in db.RangeIter(include_value=False) ])

def check_images_to_compute( settings, path_db ):
    """
    Function to check images to compute given an existing leveldb

    Ret: keyframes and labels (horitzontal/vertical)
    """

    # get all keyframes from settings structure
    keyframes, labels = get_all_keyframes( settings )

    print len(keyframes), keyframes[0]

    # output keyframe to compute
    keyframes_done = []

    # ensure that we only take filename without path
    keyframes_filename_total = map( os.path.basename, keyframes )
    print len(keyframes_filename_total), keyframes_filename_total[0]

    # build dictionary with full path to the images
    full_path = {}
    label_dic = {}
    for key, full, label in zip(keyframes_filename_total, keyframes, labels):
        full_path[key] = full
        label_dic[key] = label
        print full

    count = 0

    if not os.path.exists( path_db ):
        # Generate assignments
        print "Computing from scratch!"
        return np.array(keyframes), np.array(labels)

    else:
        print "Assignments db existing."
        # Init levelDB
        db = leveldb.LevelDB( path_db )

        # Check the keyframes already computed
        keyframes_done = []
        count = 1
        for key in db.RangeIter(include_value=False):
            a = key.split('@')[0]
            keyframes_done.append( a )
            count += 1
            if count % 100 == 0:
                print a, key, full_path[a]
                print "{} of {}".format( count, len(keyframes) )

        if len(keyframes_done)==len(keyframes_filename_total):
            print "Assignments completed!"
            return None, None
        else:
            print len(keyframes_filename_total), len(keyframes_done)
            keyframes_todo=set(keyframes_filename_total)-set(keyframes_done)

        # rebuild list to compute =
        final_k = []
        final_l = []
        for key in keyframes_todo:
            final_k.append( full_path[key] )
            final_l.append( label_dic[key] )
	del db
        print
        print "Finishing assingments. {} images to compute.".format(len(final_k))
        return np.array(final_k), np.array(final_l)

def get_from_not_read_txt( filename ):
    data = np.loadtxt( open( filename , 'r' ), dtype='str' )
    keyframes = data[:,1]

    b = []
    for a in data[:,2]:
        if a == "True":
            b.append(True)
        else:
            b.append(False)
    labels = np.array( b )

    a = []
    for k in keyframes:
        a.append( str("/media/eva/Eva "+k) )
    keyframes = np.array(a)

    return keyframes, labels


if __name__=="__main__":
    settings = load_settings( path_settings )
    keyframes, labels = check_images_to_compute( settings, path_db )

    print len(keyframes)
