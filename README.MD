# Bags of Local Convolutional Features for Scalable Instance Search

| ![Logo Icmr][icmrlogo] | [Best Poster Award](http://www.icmr2016.com/awards.html) at the [ACM International Conference on Multimedia Retrieval (ICMR) 2016](http://www.icmr2016.com/index.html)   |
|:-:|---|

[icmrlogo]: https://raw.githubusercontent.com/evamohe/BoW_CNN_InstanceSearch/master/authors/logo_1.jpg?token=AHPpwJEFJz9ZB5Q5RyZ_O6HuU7_S2yp-ks5X1vMgwA%3D%3D

| ![Eva Mohedano][EvaMohedano-photo] | ![Amaia Salvador][AmaiaSalvador-photo] | ![Kevin McGuinness][KevinMcGuinness-photo] |  ![Xavier Giro-i-Nieto][XavierGiro-photo] | ![Noel O'Connor][NoelOConnor-photo]  | ![Ferran Marques][FerranMarques-photo]|
|:-:|:-:|:-:|:-:|:-:|:-:|
| [Eva Mohedano][EvaMohedano-web]  | [Amaia Salvador][AmaiaSalvador-photo] | [Kevin McGuinness][KevinMcGuinness-web]   | [Xavier Giro-i-Nieto][XavierGiro-web]   | [Noel O'Connor][NoelOConnor-web]   | [ Ferran Marques][FerranMarques-web]  |

[EvaMohedano-web]: https://www.insight-centre.org/users/eva-mohedano
[AmaiaSalvador-web]: https://imatge.upc.edu/web/people/amaia-salvador
[KevinMcGuinness-web]: https://www.insight-centre.org/users/kevin-mcguinness
[XavierGiro-web]: https://imatge.upc.edu/web/people/xavier-giro
[NoelOConnor-web]: https://www.insight-centre.org/users/noel-oconnor
[FerranMarques-web]: https://imatge.upc.edu/web/people/ferran-marques


[EvaMohedano-photo]: https://raw.githubusercontent.com/imatge-upc/retrieval-2016-lostobject/master/authors/Eva.jpg?token=AKsMd4iuttxHH44mYL3mPpJEtSvXVXF8ks5Xe-AWwA%3D%3D "Eva Mohedano"
[AmaiaSalvador-photo]: https://raw.githubusercontent.com/evamohe/BoW_CNN_InstanceSearch/master/authors/salvador.jpg?token=AHPpwCjeOzq3duY0lkLTXEe8H7giXJEPks5X1vG2wA%3D%3D "Amaia Salvador" 
[KevinMcGuinness-photo]: https://raw.githubusercontent.com/imatge-upc/retrieval-2016-lostobject/master/authors/Kevin.jpg?token=AKsMd4VU31T7Bh8CztufWEWNudazbB_Uks5Xe-AxwA%3D%3D "Kevin McGuinness"
[XavierGiro-photo]: https://raw.githubusercontent.com/evamohe/BoW_CNN_InstanceSearch/master/authors/giro.jpg?token=AHPpwDdVdPYfMIwMBgHbjK9pPMJva1GOks5X1vHIwA%3D%3D "Xavier Giro-i-Nieto"
[NoelOConnor-photo]: https://raw.githubusercontent.com/imatge-upc/retrieval-2016-lostobject/master/authors/Noel.jpg?token=AKsMdyemO5eJke9B9rqdRtA7otJscq1wks5Xe-BEwA%3D%3D "Noel O'Connor"
[FerranMarques-photo]: https://raw.githubusercontent.com/evamohe/BoW_CNN_InstanceSearch/master/authors/marques.jpg?token=AHPpwN3NJU0PqS8i-r2Ng7TpRjlQV22vks5X1vHWwA%3D%3D

A joint collaboration between:

| ![logo-insight] | ![logo-dcu] | ![logo-upc] | ![logo-etsetb] | ![logo-gpi] | 
|:-:|:-:|:-:|:-:|:-:|
| [Insight Centre for Data Analytics][insight-web] | [Dublin City University (DCU)][dcu-web]  |[Universitat Politecnica de Catalunya (UPC)][upc-web]   | [UPC ETSETB TelecomBCN][etsetb-web]  | [UPC Image Processing Group][gpi-web] | 

[insight-web]: https://www.insight-centre.org/ 
[dcu-web]: http://www.dcu.ie/
[upc-web]: http://www.upc.edu/?set_language=en 
[etsetb-web]: https://www.etsetb.upc.edu/en/ 
[gpi-web]: https://imatge.upc.edu/web/ 


[logo-insight]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/insight.jpg "Insight Centre for Data Analytics"
[logo-dcu]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/dcu.png "Dublin City University"
[logo-upc]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/upc.jpg "Universitat Politecnica de Catalunya"
[logo-etsetb]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/etsetb.png "ETSETB TelecomBCN"
[logo-gpi]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/gpi.png "UPC Image Processing Group"

## Abstract 
This work proposes a simple instance retrieval pipeline based on encoding the convolutional features of CNN using the bag of words aggregation scheme (BoW). Assigning each local array of activations in a convolutional layer to a visual word produces an assignment map, a compact representation that relates regions of an image with a visual word. We use the assignment map for fast spatial reranking, obtaining object localizations that are used for query expansion. We demonstrate the suitability of the BoW representation based on local CNN features for instance retrieval, achieving competitive performance on the Oxford and Paris buildings benchmarks. We show that our proposed system for CNN feature aggregation with BoW outperforms state-of-the-art techniques using sum pooling at a subset of the challenging TRECVid INS benchmark.


## Publication

Find our paper at [ACM Digital Library](http://dx.doi.org/10.1145/2911996.2912061), [arXiv](https://arxiv.org/abs/1604.04653) and [DCU Doras](http://doras.dcu.ie/21175/).

![Image of the paper](https://raw.githubusercontent.com/evamohe/BoW_CNN_InstanceSearch/master/docs/image.jpeg?token=AHPpwMnNlL9yWB3Lt7588wSrlzim1tO4ks5X1vazwA%3D%3D)

Please cite with the following Bibtex code:

````
@inproceedings{Mohedano:2016:BLC:2911996.2912061,
 author = {Mohedano, Eva and McGuinness, Kevin and O'Connor, Noel E. and Salvador, Amaia and Marques, Ferran and Giro-i-Nieto, Xavier},
 title = {Bags of Local Convolutional Features for Scalable Instance Search},
 booktitle = {Proceedings of the 2016 ACM on International Conference on Multimedia Retrieval},
 series = {ICMR '16},
 year = {2016},
 isbn = {978-1-4503-4359-6},
 location = {New York, New York, USA},
 pages = {327--331},
 numpages = {5},
 url = {http://doi.acm.org/10.1145/2911996.2912061},
 doi = {10.1145/2911996.2912061},
 acmid = {2912061},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {bag of words, convolutional neural networks, instance retrieval},
} 
````

## Best Poster Award at ICMR 2016

![https://github.com/imatge-upc/retrieval-2016-icmr/raw/master/docs/icmr2016-poster.pdf](https://github.com/imatge-upc/retrieval-2016-icmr/blob/master/docs/poster700x583.png?raw=true)

![Poster presenters](https://github.com/imatge-upc/retrieval-2016-icmr/blob/master/authors/PosterPresenters.jpg?raw=true)


## Code Instructions

### Description

It contains scripts to build Bag of Visual Words based on local CNN features to perform instance search in three different datasets:

* [Oxford Buildings](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) (and Oxford 105k).

* [Paris Buildings](http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/) (and Paris 106k).

* Trecvid_subset: Subset of 23.614 keyframes/~13.000video shots from [TRECVid-INS](http://www-nlpir.nist.gov/projects/tv2015/) dataset. Keyframes extracted uniformly at 1/4fps. Queries and groundtruth correspond to [INS2013](http://www-nlpir.nist.gov/projects/tv2013/tv2013.html).


### Prerequisits

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


### How to run it?

*bow_pipeline* folder contain the main scripts. Parameters must be specified in the *settings.json* file located in the folder named as the dataset.


#### Step 1: FEATURE EXTRACTION (bow_pipeline/A_feature_extraction.py)

This script extracts features from a pre-trained CNN (by default the fully convolutional [VGG-16 network](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md). It computers descriptors from the specified layer/s *Layer_output* parameter in a levelDB dataset in *featuresDB* paramenter in the *settings.json*, with the format [*featuresDB*]/[*layer*]_db.

**NOTE**
Features are stored as the original dictionary created by the *Net* class from [caffe](http://caffe.berkeleyvision.org/). For reading, you should use the class *Local_Feature_ReaderDB* located in *bow_pipeline/reader.py*. This class contains methods to extract local features in the format (*n_samples*, *n_dimensions*)/image for BoW encoding. It also performs SumPooling, generating (*1*, *n_dimensions*)/image. It also contains methods to interpolate feature maps when reading. For more info in, check *bow_pipeline/reader.py* script.


#### Step 2: BUILDING VISUAL VOCABULARY (bow_pipeline/B_processing_clustering.py)

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


#### Step 3: ASSIGNMENTS AND INVERTED FILE GENERATION (bow_pipeline/C_bow_representation.py)

Once the visual vocabulary is build, we can compute the assignments based on the local features/image and we can index the dataset of images. It is necessary to set the following parameters (check the script):

```
settings file for the dataset
dim_input -- network input dimensions in string format
network="vgg16" -- string with network name to use
list_layers -- list with layer/s to use
new_dim -- tuple with the feature map dimension
```


#### Step 4: RANKINGS GENERATION (bow_pipeline/D_rankings_BoW.py)

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


#### Step 5: EVALUATION (bow_pipeline/evaluate_[Oxf_Par/trecvid].py)

It computes mean average precission for rankings generated in Step 4.

* *evaluate_Oxf_Par* for Paris and Oxford datasets

* evaluate_trecvid for TRECVid subset.


**NOTE** It generate *map.txt* in bow_pipeline folder with results.
