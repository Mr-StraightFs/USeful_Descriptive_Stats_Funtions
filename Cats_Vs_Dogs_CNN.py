import os
import zipfile
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf


#First Use the Following commented Code to fetch your data from the internet and extract it locally
# wget --no-check-certificate \
#   https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
#   -O /tmp/cats_and_dogs_filtered.zip
# local_zip = '/tmp/cats_and_dogs_filtered.zip'
#
# zip_ref = zipfile.ZipFile(local_zip, 'r')
#
# zip_ref.extractall('/tmp')
# zip_ref.close()

# STep2 : Define the training and valivation subdirectories
# base_dir = '/tmp/cats_and_dogs_filtered'
#
# train_dir = os.path.join(base_dir, 'train')
# validation_dir = os.path.join(base_dir, 'validation')
#
# # Directory with our training cat/dog pictures
# train_cats_dir = os.path.join(train_dir, 'cats')
# train_dogs_dir = os.path.join(train_dir, 'dogs')
#
# # Directory with our validation cat/dog pictures
# validation_cats_dir = os.path.join(validation_dir, 'cats')
# validation_dogs_dir = os.path.join(validation_dir, 'dogs')


##printing the Total number of images
# print('total training cat images :', len(os.listdir(      train_cats_dir ) ))
# print('total training dog images :', len(os.listdir(      train_dogs_dir ) ))
#
# print('total validation cat images :', len(os.listdir( validation_cats_dir ) ))
# print('total validation dog images :', len(os.listdir( validation_dogs_dir ) ))

