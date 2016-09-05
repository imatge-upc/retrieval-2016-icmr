# imports
import os, glob, shutil
import caffe

# set paths to dataset images
path_images = "/media/eva/Eva Data/Datasets/Paris_dataset/paris"
path_images_all = "/media/eva/Eva Data/Datasets/Paris_dataset/images"

if not os.path.isdir( path_images_all ):
    os.makedirs( path_images_all )



# dataset topics
topics = os.listdir( path_images )

# image counters
count = 0
v_c = 0
h_c = 0
corrupted = 0


# index aspect ratio
txt_vertical = 'list_paris_vertical.txt'
txt_horitzontal = 'list_paris_horitzontal.txt'
txt_corrupt = 'corrupted.txt'

f_vert = open( txt_vertical, "w" )
f_hort = open( txt_horitzontal, "w" )
f_corr = open( txt_corrupt, "w" )



for topic in topics:

    # get images per topic
    images = os.listdir( os.path.join( path_images, topic ) )

    for name_ima in images:

        # check vertical/horitzontal
        try:
            ima = caffe.io.load_image( os.path.join( path_images, topic, name_ima ) )

            # check panorama/portrait
            if ima.shape[0] > ima.shape[1]:
                # portrait -- vertical
                f_vert.write( "{}\n".format( os.path.join( path_images_all, name_ima ) ) )
                v_c += 1
            else:
                # panorama -- horitzontal
                f_hort.write( "{}\n".format( os.path.join( path_images_all, name_ima ) ) )
                h_c += 1


            # copy images into general folder
            shutil.copyfile( os.path.join( path_images, topic, name_ima ), os.path.join( path_images_all, name_ima ) )
            count += 1
        except:
            f_corr.write( "{}\n".format( name_ima ) )
            corrupted += 1

f_vert.close()
f_hort.close()
f_corr.close()

print "Total images {}".format( count )
print "\t vertical: {}".format( v_c )
print "\t horitzontal: {}".format( h_c )
