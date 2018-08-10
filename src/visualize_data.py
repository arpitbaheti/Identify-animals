import numpy as np
from matplotlib import pyplot
from glob import glob
import os

class_count = []
folder = 'data/train_90+new/'
for root, dirnames, filenames in os.walk(folder):
    for dir in dirnames:
        folder_t = folder+dir
        files = glob(folder_t+'/*.jp*g')
        class_count.append(len(files))
    break

pyplot.bar(np.arange(len(class_count)), class_count, align='center')
pyplot.xlabel('Class')
pyplot.ylabel('Number of training examples')
pyplot.xlim([-1, len(class_count)])
pyplot.savefig('train_data_aug.png')
