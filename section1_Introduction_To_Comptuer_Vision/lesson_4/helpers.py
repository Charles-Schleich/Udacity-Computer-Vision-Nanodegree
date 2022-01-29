import os 
import glob 
import matplotlib.image as mpimg

import cv2 


def load_dataset(image_dir):
    im_list = []
    image_types = ["day","night"]
    
    # Iterate through each type folder
    for im_type in image_types:

        # iterate through each image file in each image_type folder
        # glib reads in  any image with the extension "image_dir/im_type/*"

        for file in glob.glob(os.path.join(image_dir, im_type,"*")):
            # read in image

            im = mpimg.imread(file)
            
            #
            if not im is None:
                im_list.append((im,im_type))
    
    return im_list

def standardize_input(image):
    standard_im = cv2.resize(image,(1100,600))
    return standard_im

def encode(label):
    numerical_val=0
    if(label== 'day'):
        numerical_val=1
    return numerical_val

def standardize(image_list):
    
    standard_list=[]

    for item in image_list:
        image = item[0]
        label = item[1]

        standardized_image = standardize_input(image)
        binary_label = encode(label)

        standard_list.append((standardized_image,binary_label))
    return standard_list















