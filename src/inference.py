from keras import applications
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
import importlib
import sys
import os
from glob import glob
import re
from tqdm import trange

from PIL import Image, ImageOps

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Define constants
test_data_dir = 'data/test/'
test_file = 'data/test_gt.csv'
nrof_classes = 30
pred_file = 'result.csv'
img_width, img_height = 800, 800

def resize_image(im_pth):
    im = Image.open(im_pth)
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

    return new_im

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

pred_file = open(pred_file, 'w')
index_to_class = {}
index = 0
pred_file.write('image_id,')
for root, dirnames, filenames in os.walk('data/train_new'):
    dirnames = sorted(dirnames)
    for dir in dirnames:
        if index == 29:
            pred_file.write(dir)
        else:
            pred_file.write(dir+',')
        index_to_class[index] = dir
        index += 1
    break

pred_file.write('\n')

# Network architecture
model_def = 'models.network'
arch = sys.argv[1]
network = importlib.import_module(model_def, 'inference')

# load model
model, base_model = network.inference(nrof_classes, arch)
print('Model loaded.')

model.load_weights(sys.argv[2])

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])

files = natural_sort(glob(test_data_dir + '/*.jpg'))

desc = 'inferencing'
bar = trange(len(files), desc=desc, leave=True, ncols=120, ascii=True)
test_gt = open(test_file, 'w')
for k in bar:
    img_path = files[k]
    image_file = img_path.split('/')[-1:]
    img = image.load_img(img_path, target_size=(800, 800))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)

    import pdb;pdb.set_trace()
    pred_file.write(image_file[0]+',')
    for i in range(30):
        if i == 29:
            pred_file.write(str(preds[0][i])+'\n')
        else:
            pred_file.write(str(preds[0][i])+',')

    top_indice = preds[0].argsort()[-1:][::-1]
    test_gt.write(image_file[0] +","+ index_to_class[top_indice[0]]+'\n')

test_gt.close()
pred_file.close()
