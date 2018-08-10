from __future__ import absolute_import
from __future__ import print_function

from PIL import Image, ImageOps
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import optimizers
from keras.preprocessing import image
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard

import importlib
import sys
import os
from datetime import datetime
import argparse
import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define constants
img_width, img_height = 800, 800
train_data_dir = 'data/train_new'
validation_data_dir = 'data/valid_10'
nb_train_samples = 18102
nb_validation_samples = 2060
nrof_classes = 30
model_def = 'models.network'

def main(args):

    network = importlib.import_module(model_def, 'inference')

    logs_base_dir = args.logs_base_dir + '/' + args.model
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)


    # checkpoint to save the model weights.
    filepath = log_dir + "/weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    #tensor_board = TensorBoard(log_dir, histogram_freq=5, batch_size=args.batch_size, write_images=True)
    callbacks_list = [checkpoint]

    # load model
    model, base_model = network.inference(nrof_classes,
                                args.model,
                                args.keep_probability,
                                args.weight_decay)

    print('Log directory: %s' % log_dir)

    if args.pretrained_model:
        print('Pre-trained model: %s' % os.path.expanduser(args.pretrained_model))
        model.load_weights(args.pretrained_model)
        print('Pretrained model with weights loaded.')
    else:
        print('Model loaded.')

    print('Network used: %s'%args.model)
    print('Input Image size: %dx%d'%(img_width, img_height))
    print('Batch Size: %d'% args.batch_size)
    print('learning rate: %f'%args.learning_rate)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = args.train_layer

    optimizer_ = optimizer(args)

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=optimizer_, loss='categorical_crossentropy',
                                  metrics=['accuracy'])

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
                rotation_range=5,
                #width_shift_range=0.2,
                #height_shift_range=0.2,
                #shear_range=0.2,
                zoom_range=0.2,
                #preprocessing_function=resize_image,
                horizontal_flip=True,
                rescale=1. / 255)

    test_datagen = ImageDataGenerator(rescale=1. / 255)#, preprocessing_function=resize_image)

    train_generator = train_datagen.flow_from_directory(
                train_data_dir,
                target_size=(img_height, img_width),
                batch_size=args.batch_size,
                class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
                validation_data_dir,
                target_size=(img_height, img_width),
                batch_size=args.batch_size,
                class_mode='categorical')


    # fine-tune the model
    model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples/args.batch_size,
                epochs=args.epochs,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples/args.batch_size,
                callbacks=callbacks_list)

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

    im2arr = np.array(new_im, dtype=np.float32)
    return im2arr

def optimizer(args):

    if args.optimizer=='ADAGRAD':
        opt = optimizers.Adagrad(args.learning_rate, decay=args.learning_rate_decay_factor)
    elif args.optimizer=='ADADELTA':
        opt = optimizers.Adadelta(args.learning_rate, decay=args.learning_rate_decay_factor)
    elif args.optimizer=='ADAM':
        opt = optimizers.Adam(args.learning_rate, decay=args.learning_rate_decay_factor)
    elif args.optimizer=='RMSPROP':
        opt = optimizers.RMSprop(args.learning_rate, decay=args.learning_rate_decay_factor, epsilon=1.0)
    elif args.optimizer=='ADAMAX':
        opt = optimizers.Adamax(args.learning_rate, decay=args.learning_rate_decay_factor)
    elif args.optimizer=='SGD':
        opt = optimizers.SGD(args.learning_rate, decay=args.learning_rate_decay_factor, momentum=0.9,  nesterov=True)
    else:
        raise ValueError('Invalid optimization algorithm')

    return opt


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--logs_base_dir', type=str,
        help='Directory where to write event logs.', default='logs/')
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    parser.add_argument('--model', type=str, choices=['InceptionResNetV2','VGG16','VGG19','ResNet50','InceptionResNetV2','MobileNet','DenseNet121'],
        help='Model definition.', default='InceptionResNetV2')
    parser.add_argument('--epochs', type=int,
        help='Number of epochs to run.', default=50)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=32)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=568)
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'SGD'],
        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate.', default=0.001)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=2)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=0.0)
    parser.add_argument('--train_layer', type=bool,
        help='Choose to train the pretrained models layers.', default=False)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

