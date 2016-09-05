#!/usr/bin/env python
from A_feature_extraction import *
import leveldb
from sklearn.preprocessing import normalize
# add utils.py
sys.path.insert(0, os.path.join(os.getcwd(),'..'))
from utils import *
from IPython import embed

#Logger
log = set_logger("reader", level_logger="info",format_logger="%(asctime)s- %(name)s - %(levelname)s - %(message)s")

# Define class to read local features
class Local_Feature_ReaderDB():
    def __init__(self, pathDB, list_layers, l2norm=True, applyPCA=None):

        # db object store in a dictionary
        self.db = {}
        self.list_layers = list_layers

        for layer in list_layers:
            self.db[layer] = leveldb.LevelDB( os.path.join(pathDB, layer+"_db") , create_if_missing=False )

        self.l2norm=l2norm

        self.pca_model=None
        if applyPCA!=None:
            if os.path.isfile( applyPCA ):
                self.pca_model = load_obj( applyPCA )
            else:
                self.pca_model = pca_model
        else:
            print applyPCA, " NO PCA."
            raw_input('PCA not loaded!! Re-lauch the script. Press key')

        self.map_dimensions_layer = self.get_dimensions()

    def build_key_str(self, dim ):
        return "@{}@{}@{}@".format( dim[0], dim[1], dim[2] )

    def get_dimensions(self):
        """
        Get default dimensions for reader.
        (vertical)
        """
        map_dimensions_layer={}

        print "Getting dim layers..."
        for layer in self.db.keys():
            # for each layer, get the firts key...
            for id_image in self.db[layer].RangeIter( include_value=False ):
                dim = (int(id_image.split('@')[1]), int(id_image.split('@')[2]), int(id_image.split('@')[3]))
                if dim[1] < dim[2]:
                    dim = ( dim[0], dim[2], dim[1] )

                break

            map_dimensions_layer[layer]=dim

            print "layer {}, dims {}".format( layer, map_dimensions_layer[layer] )

        return map_dimensions_layer

    def get_feature_maps(self,  key, vertical=True, new_dim=None, interpolation=1 ):
        '''
        Read natural features on the desired shape -- Not normalization in this step!
        '''
        maps = {}

        # iterate over all layers...
        for layer in self.list_layers:
            # READING FROM LEVELDB -----------------------------------------------------------------
            # get dimensions from key- so we are able to read maps
            dim=self.map_dimensions_layer[layer]

            if not vertical:
                dim = ( dim[0], dim[2], dim[1] )

            # If new_dim specified and we have an horitzontal image...
            if not vertical and new_dim is not None:
                new_dim = (new_dim[1], new_dim[0])

            # get extension to access to levelDB entry [key]@maps@rows@cols@
            extr=self.build_key_str( dim )

            #for i, k in  enumerate(self.db[layer].RangeIter( include_value=False )):
            #    print i, k

            # read features
            maps[layer] = np.fromstring( self.db[layer].Get( os.path.basename(key)+extr ), dtype=np.float32 ).reshape( dim )

            # RESHAPING MAPS -----------------------------------------------------------------------
            # if we've set new dimensions...
            if new_dim is not None and new_dim!=dim[1:]:
                log.debug("New DIM: Interpolating {}, {}".format(interpolation, new_dim))
                #RESHAPE -- check reshape_maps / reshape_maps_zoom
                t0 = time.time()
                if interpolation == 0:
                    maps[layer] = reshape_maps(maps[layer], new_dim)
                elif interpolation==1 or interpolation==2:
                    maps[layer] = reshape_maps_zoom(maps[layer], new_dim, interpolation)
                t1 = time.time()

            # else, set all layers to the same dim! (first one)
            else:
                # reshape rest of layers to the first layer dim in the list
                new_dim = maps[self.list_layers[0]].shape[1:]
                #RESHAPE -- check reshape_maps / reshape_maps_zoom
                t0 = time.time()
                if maps[layer].shape[1:] !=new_dim:
                    log.debug("Default DIM: Interpolating {} ".format(interpolation))
                    if interpolation == 0:
                        maps[layer] = reshape_maps(maps[layer], new_dim)
                elif interpolation==1 or interpolation==2:
                    maps[layer] = reshape_maps_zoom(maps[layer], new_dim, interpolation)
                t1 = time.time()

            log.debug("Layer {} ; {}; elapsed {}".format( layer, maps[layer].shape, t1-t0 ))

            # we return all features resized to an unique dimension (so we can contatenate them :-) )
        return maps

    def get_features(self, key, vertical=True, new_dim=None, interpolation=1, n_samples=None ):
        '''
        Reshape the features into (n_samples, n_dim) and apply l2-normalization
        and concatenate layers (increase n_dim)
        '''

        #dictionary with maps per layer - UNIQUE SIZE
        t0 = time.time()
        maps = self.get_feature_maps( key, vertical, new_dim=new_dim, interpolation=interpolation )
        t1 = time.time()
        log.debug("maps loaded! : elapsed {}".format( t1-t0))

        # If we only have one layer to process...
        #extract info of the first layer in the list
        dim = maps[self.list_layers[0]].shape
        t0=time.time()
        maps[self.list_layers[0]] = np.swapaxes(maps[self.list_layers[0]], 0,1).swapaxes(1,2)
        features = np.reshape(maps[self.list_layers[0]], (dim[1]*dim[2], dim[0]))
        t1=time.time()
        log.debug("reshaping n_samples, n_dim : elapsed {}".format( t1-t0 ))


        #if we read more than one layer then concatenate activations
        if len( self.list_layers )>1:
            for layer in self.list_layers[1:]:
                maps[layer] = np.swapaxes(maps[layer], 0,1).swapaxes(1,2)
                features_ = np.reshape(maps[layer], (dim[1]*dim[2], dim[0]))
                features = np.concatenate( (features, features_), axis=1 )
            log.debug("Concatenation done! {}".format(features.shape[1]))

        # normalize features if requiered
        if self.l2norm:
            t0=time.time()
            features = normalize(features)
            t1=time.time()
            log.debug("feats normalized! : elapsed {}".format( t1-t0) )

        # apply PCA for dim reduction
        if self.pca_model!=None:
            t0=time.time()
            features = self.pca_model.transform(features)
            t1=time.time()
            #re-normalize features!
            if self.l2norm:
                features = normalize(features)
            log.debug("feats with PCA! : elapsed {}".format( t1-t0) )

        log.debug("Dimensions features {}".format(features.shape) )

        if n_samples is not None:
            # make a feature index
            idx=np.arange(features.shape[0])
            # shuffle it
            np.random.shuffle(idx)
            # return n_samples after shuffling
            return features[idx[:n_samples], :]
        else:
            # we return processed features (l2 and PCA)
            return features

    def pool_features(self, key, vertical=True, new_dim=None, pooling='sum', interpolation=1 ):
        '''
        Perform pooling (sum average on the feature maps)
        '''
        #dictionary with maps per layer - UNIQUE SIZE
        t0 = time.time()
        maps = self.get_feature_maps( key, vertical, new_dim=new_dim, interpolation=interpolation )
        t1 = time.time()
        log.debug("maps loaded! : elapsed {}".format( t1-t0))

        # If we only have one layer to process...
        #extract info of the first layer in the list
        dim = maps[self.list_layers[0]].shape
        t0=time.time()

        # normalize feature
        if pooling == 'sum':
            features =  np.sum( maps[ self.list_layers[0] ], axis=1 ).sum(axis=1)

            t1=time.time()
            log.debug("Pooling features {} : shape {}".format( features.shape, t1-t0 ))


        elif pooling == 'max':
            features =  np.max( maps[ self.list_layers[0] ], axis=1 ).max(axis=1)

            t1=time.time()
            log.debug("Pooling features {} : shape {}".format( features.shape, t1-t0 ))

        #if we read more than one layer then concatenate activations
        if len( self.list_layers )>1:
            for layer in self.list_layers[1:]:
                if pooling == 'sum':
                    features = np.concatenate( (features, np.sum( maps[ self.list_layers[0] ], axis=1 ).sum(axis=1) ) )
                elif poolint == 'max':
                    features = np.concatenate( (features, np.max( maps[ self.list_layers[0] ], axis=1 ).max(axis=1) ) )

            log.debug("Concatenation done! {}".format(features.shape))

        # normalize features if requiered
        if self.l2norm:
            t0=time.time()
            features = normalize(features)
            t1=time.time()
            log.debug("feats normalized! : elapsed {}".format( t1-t0) )

        # apply PCA for dim reduction
        if self.pca_model!=None:
            t0=time.time()
            features = self.pca_model.transform(features)
            t1=time.time()
            #re-normalize features!
            if self.l2norm:
                features = normalize(features)
            log.debug("feats with PCA! : elapsed {}".format( t1-t0) )

        log.debug("Dimensions features {}".format(features.shape) )

        return features.squeeze()

    @classmethod
    def from_settings(cls, settings):
        # build feature path for reading
        pathDB = os.path.join(str(settings["featuresDB"]),
        str(settings["Feature_extractor"]),
        str(settings["input_size"][2])+"_"+str(settings["input_size"][3]))

        #get layers to combine
        layers = settings["Layer_output"]

        return cls(pathDB, layers)

    @property
    def get_dimension(self):
        keys = self.map_dimensions_layer.keys()

        for key in keys:
            print "layer {}, dimensions {}".format( key, self.map_dimensions_layer[key] )
        return self.map_dimensions_layer[keys[0]]


if __name__ == "__main__":
    settings=load_settings("/home/eva/Workspace/icmr_pipeline/oxford/settings.json")
    reader = Local_Feature_ReaderDB.from_settings(settings)
    print reader.get_dimension
