import glob
import os
import sys
import math
from tqdm import tqdm

train_file = 'data/valid_100.csv'
valid_file = 'data/valid_0.csv'
data_file  = 'data/valid_gt.csv'

data = open(data_file, 'r').readlines()

train_csv = open(train_file, 'w')
valid_csv = open(valid_file, 'w')

total = len(data)
val = int(math.ceil(total * 0.0))
training = total - val

for i in tqdm(range(total)):
    if i == 0:
        continue
    if (i <= training):
        path = 'data/valid/' + data[i]
        train_csv.write(path)
    else:
        path = 'data/valid/' + data[i]
        valid_csv.write(path)

train_csv.close()
valid_csv.close()

train_csv = open(train_file, 'r').readlines()
valid_csv = open(valid_file, 'r').readlines()

# Destination for Train dataset
train  = sys.argv[1]
# Destination for Valid dataset
valid = sys.argv[2]

if not os.path.exists(valid):
    os.makedirs(valid)
if not os.path.exists(train):
    os.makedirs(train)

for i in tqdm(range(len(train_csv))):
    category = train_csv[i].split(',')[1][:-1]

    source = train_csv[i].split(',')[0]
    destination = train + '/' + category

    if not os.path.exists(destination):
        os.makedirs(destination)

    cp_cmd = 'cp ' + source + ' ' + destination +'/'
    os.system(cp_cmd)

for i in tqdm(range(len(valid_csv))):
    category = valid_csv[i].split(',')[1][:-1]

    source = valid_csv[i].split(',')[0]
    destination = valid + '/' + category

    if not os.path.exists(destination):
        os.makedirs(destination)

    cp_cmd = 'cp ' + source + ' ' + destination +'/'
    os.system(cp_cmd)

