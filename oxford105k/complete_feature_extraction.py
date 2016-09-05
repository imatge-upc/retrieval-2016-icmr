import sys
sys.path.insert(0,'/home/eva/Workspace/icmr_pipeline/bow_pipeline')
from A_feature_extraction import *




if __name__ == "__main__":

    # oxford105k subset dataset
    settings = load_settings( 'settings.json' )

    # levelDB storer object
    storer = LeveldbStorer.from_settings(settings)

    count=0
    for a in storer.db_dict["pool5"].RangeIter(include_value=False):
        count+=1
        print "{} {}".format( count, a )
    print "total {} {}".format(count, count)
    s = raw_input("--> Press key")

    """
    # Feature extractor - [Image Dimensions fixed to 1/3 in the prototxt]
    fx = Local_Extractor.from_settings(settings)

    # Process Vertical
    keyframes = load_paths_txt( settings["keyframesVertical5k"] )

    #keyframes = [ os.path.join( settings["path_images"], os.path.basename(key) )for key in keyframes ]
    extract_features(fx, keyframes, storer)

    # Process Horitzontal
    dim = fx.input_shape
    log.debug( "dim {}".format(dim) )
    fx.reshape_input( fx.batch_size, dim[0], dim[2], dim[1] )
    keyframes = load_paths_txt( settings["keyframesHoritzontal5k"] )
    #keyframes = [ os.path.join( settings["path_images"], os.path.basename(key) )for key in keyframes ]
    extract_features(fx, keyframes, storer)
    """
