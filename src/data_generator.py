from __future__ import absolute_import
from __future__ import print_function

from PIL import Image, ImageOps
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import importlib
import sys
import os
from datetime import datetime
import argparse
from glob import glob
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

cls = sys.argv[1]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IMAGE_SIZE = 299
img_width, img_height = 299, 299
def tf_resize_images(file_path):
    X_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, (None, None, 3))
    tf_img = tf.image.resize_images(X, (IMAGE_SIZE, IMAGE_SIZE),
                                   tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        img = mpimg.imread(file_path)[:, :, :3] # Do not read alpha channel.
        resized_img = sess.run(tf_img, feed_dict = {X: img})
        return resized_img

def resize_image(im):
    im = Image.fromarray(np.uint8(im))
    old_size = im.size  # old_size[0] is in (width, height) format

    ratio = float(img_width)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])


    im = im.resize(new_size, Image.ANTIALIAS)

    new_im = Image.new("RGB", (img_width, img_height))
    new_im.paste(im, ((img_width-new_size[0])//2,
                                    (img_height-new_size[1])//2))

    delta_w = img_width - new_size[0]
    delta_h = img_height - new_size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    new_im = ImageOps.expand(im, padding)

    im2arr = np.array(new_im)
    return im2arr

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
                      shear_range=0.2,
                      zoom_range=0.2,
                      horizontal_flip=True)
                      #preprocessing_function=resize_image)

# save augmented data
folder = 'data/train_90/'+ cls +'/'
images = glob(folder + "*.jp*g")

save_folder = 'data/train_new/' + cls +'/'
if not os.path.isdir(save_folder):  # Create the log directory if it doesn't exist
    os.makedirs(save_folder)

#import pdb;pdb.set_trace()

for image_ in images:
    img = load_img(image_)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = np.expand_dims(x, axis=0)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in train_datagen.flow(x, batch_size=1,
                                           save_to_dir=save_folder, save_prefix=cls, save_format='jpg'):
        break

    files = glob(save_folder + "/*.jpg")
    print (len(files), int(sys.argv[2]), len(files) >= int(sys.argv[2]))
    #import pdb; pdb.set_trace()
    if (len(files) >= int(sys.argv[2])):
        break  # otherwise the generator would loop indefinitely

    #import pdb;pdb.set_trace()
