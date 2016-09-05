# **Bags of Local Convolutional Features for Scalable Instance Search_**
Authors: *Eva Mohedano, Amaia Salvador, Kevin McGuinnes, Xavier Giro-I-Nieto, Noel O'Connor and Ferran Marques.*





## Description ##

It contains scripts to build Bag of Visual Words based on local CNN features to perform instance search in three different datasets:

* [Oxford Buildings](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) (and Oxford 105k).

* [Paris Buildings](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) (and Paris 106k).

* Trecvid_subset: Subset of 23.614 keyframes/~13.000video shots from [TRECVid-INS](http://www-nlpir.nist.gov/projects/tv2015/) dataset. Keyframes extracted uniformly at 1/4fps. Queries and groundtruth correspond to [INS2013](http://www-nlpir.nist.gov/projects/tv2013/tv2013.html).


Prerequisits
------------
Python packages necessary specified in *requirements.txt* run:

```
 pip install -r requirements.txt

```

It also needs:

*  **[caffe](http://caffe.berkeleyvision.org/)** with python support

*  **[vlfeat](https://github.com/dougalsutherland/vlfeat-ctypes)** library
    * Once installed, modify 'kmeans.py' file, located in the vlfeat python package: (i.e /usr/local/lib/python2.7/dist-packages/vlfeat_ctypes-0.1.4-py2.7.egg/vlfeat/kmeans.py)
      for lib/kmeans.py of this repo.

*  **invidx** module. Follow instructions in 'lib/py=inverted-index'/

**NOTE**
You can create a virtual enviroment to set the specific dependences of this project in an independent enviroment, without modifying your original python enviroment. Check [how to create a virtual enviroment](http://docs.python-guide.org/en/latest/dev/virtualenvs/). Using *--system-site-packages* flag when creating the venv will copy all the packages from your python installation. This will prevent you of re-installing packages in the new virtual enviroment that you have already installed. This will allow you to install only the new packages.


## How to run it?

*bow_pipeline* folder contain the main scripts. Parameters must be specified in the *settings.json* file located in the folder named as the dataset.


### Step 1: FEATURE EXTRACTION (bow_pipeline/A_feature_extraction.py)

This script extracts features from a pre-trained CNN (by default the fully convolutional [VGG-16 network](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md). It computers descriptors from the specified layer/s *Layer_output* parameter in a levelDB dataset in *featuresDB* paramenter in the *settings.json*, with the format [*featuresDB*]/[*layer*]_db.

**NOTE**
Features are stored as the original dictionary created by the *Net* class from [caffe](http://caffe.berkeleyvision.org/). For reading, you should use the class *Local_Feature_ReaderDB* located in *bow_pipeline/reader.py*. This class contains methods to extract local features in the format (*n_samples*, *n_dimensions*)/image for BoW encoding. It also performs SumPooling, generating (*1*, *n_dimensions*)/image. It also contains methods to interpolate feature maps when reading. For more info in, check *bow_pipeline/reader.py* script.


### Step 2: BUILDING VISUAL VOCABULARY (bow_pipeline/B_processing_clustering.py)

It performs the clustering of the local features. It is necessary to set the
following parameters:

```
TRAIN_PCA=True -- If we want PCA/whitenning (default true)
TRAIN_CENTROIDS=True -- if we want to train centroids (default true)
l2norm=True -- if we want to perform l2-norm on the features (default true)
n_centers=25000 -- # of clusters
pca_dim=512 -- dimensions when doing PCA
```

It is necessary to specify the settings.json (different for each dataset, which contains paths for reading features/store models.


### Step 3: ASSIGNMENTS AND INVERTED FILE GENERATION (bow_pipeline/C_bow_representation.py)

Once the visual vocabulary is build, we can compute the assignments based on the local features/image and we can index the dataset of images. It is necessary to set the following parameters (check the script):

```
settings file for the dataset
dim_input -- network input dimensions in string format
network="vgg16" -- string with network name to use
list_layers -- list with layer/s to use
new_dim -- tuple with the feature map dimension
```


### Step 4: RANKINGS GENERATION (bow_pipeline/D_rankings_BoW.py)

It uses the inverted file generated and generates the rankigns for the queries. It computes the assignments for the query on-the-fly. It is necessary to set parameters as in Step 3, with the additional:

```
  	- masking = 3 (Kind of masking applied on query):
        0== No maks;
        1 == Only consider words from foreground;
        2 == Only consider words from background [CHANGED];
        3 == Apply inverse weight to the foreground object
  	- augmentation = [0] [TO REVIEW]
        0 == No augmentation;
        1 == 0+Flipped image;
        2 == 0+Zoomed image (same size as input net, but just capturing center crop
             of the image when it has been zoomed to the double size).
        3 == 0+Flipped zoomed crop
    - QUERY_EXPANSION [TO_ADD]
```

**NOTE**
If using bow_pipeline/D_rankings_Pooling.py; Skip steps 2 and 3. The whole pipeline is based in sumpooled features (without inverted index and codebook generation).

*.txt* files are generated per query under *datasetFolder/lists_[bow/pooling]*.


### Step 5: EVALUATION (bow_pipeline/evaluate_[Oxf_Par/trecvid].py)

It computes mean average precission for rankings generated in Step 4.

* *evaluate_Oxf_Par* for Paris and Oxford datasets

* evaluate_trecvid for TRECVid subset.


**NOTE** It generate *map.txt* in bow_pipeline folder with results.
