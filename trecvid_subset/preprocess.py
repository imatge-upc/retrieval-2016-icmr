# imports
import os, glob, shutil
import caffe

path_images = "/media/eva/Eva Data/Datasets/ins15/1_images/gt_imgs"


# dataset topics
shots = os.listdir( path_images )

# index aspect ratio
txt_vertical = 'list_trevcid_subset_vertical.txt'
txt_horitzontal = 'list_trecvid_subset_horitzontal.txt'
txt_corrupt = 'corrupted.txt'

f_vert = open( txt_vertical, "w" )
f_hort = open( txt_horitzontal, "w" )
f_corr = open( txt_corrupt, "w" )

# image counters
count = 0
v_c = 0
h_c = 0
corrupted = 0

for i, shot in enumerate( shots ):

    # get images per topic
    images = os.listdir( os.path.join( path_images, shot ) )

    for name_ima in images:

        # check vertical/horitzontal
        try:
            ima = caffe.io.load_image( os.path.join( path_images, shot, name_ima ) )

            # check panorama/portrait
            if ima.shape[0] > ima.shape[1]:
                # portrait -- vertical
                f_vert.write( "{}\n".format( os.path.join( path_images, shot, name_ima ) ) )
                v_c += 1
            else:
                # panorama -- horitzontal
                f_hort.write( "{}\n".format( os.path.join( path_images, shot, name_ima ) ) )
                h_c += 1

            count += 1
            print "{}/{}\tprocessing {}".format( i, len(shots), count )
        except:
            f_corr.write( "{}\n".format( name_ima ) )
            corrupted += 1


f_vert.close()
f_hort.close()
f_corr.close()

print "Total images {}".format( count )
print "\t vertical: {}".format( v_c )
print "\t horitzontal: {}".format( h_c )
