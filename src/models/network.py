from keras import applications
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing import image
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D
from keras import regularizers

def base_network(network = 'InceptionV3'):
    if network == 'InceptionV3':
        base_model = applications.InceptionV3(weights='imagenet', include_top=False)
    elif network == 'VGG16':
        base_model = applications.VGG16(weights='imagenet', include_top=False)
    elif network == 'VGG19':
        base_model = applications.VGG19(weights='imagenet', include_top=False)
    elif network == 'ResNet50':
        base_model = applications.ResNet50(weights='imagenet', include_top=False)
    elif network == 'InceptionResNetV2':
        base_model = applications.InceptionResNetV2(weights='imagenet', include_top=False)
    elif network == 'MobileNet':
        base_model = applications.MobileNet(weights='imagenet', include_top=False)
    elif network == 'DenseNet121':
        base_model = applications.DenseNet121(weights='imagenet', include_top=False)
    else:
        print ('Wrong Model selected.')
        return None

    return base_model

def inference(nrof_classes, network, keep_prob = 1.0, weight_decay=0.0):
    # build the network
    base_model = base_network(network)

    # build a classifier model to put on top of the convolutional model
    x = base_model.output

    # add a global spatial average pooling layer
    #x = GlobalAveragePooling2D()(x)

    #Dropout layer for regularization
    #x = Dropout(keep_prob)(x)

    # add a fully-connected layer
    #x = Dense(1024,
    #        activation='relu',
    #        kernel_regularizer=regularizers.l2(weight_decay))(x)

    # Dropout layer for regularization
    #x = Dropout(keep_prob)(x)

    # logistic layer -- for given classes
    predictions = Dense(nrof_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model, base_model
