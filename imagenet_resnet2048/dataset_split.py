import os
import shutil
import random

folders = os.listdir("/opt/huangyanchun/workspace/datasets/imagenet/train_picvec")
folders_filter = []
for folder in folders:
    if folder.startswith('n'):
        folders_filter.append(folder)

random.shuffle(folders_filter)
folders_filter = folders_filter[0:234]
for folder in folders_filter:
    shutil.move("/opt/huangyanchun/workspace/datasets/imagenet/train_picvec/%s" % folder, "/opt/huangyanchun/workspace/datasets/imagenet/val_picvec")
