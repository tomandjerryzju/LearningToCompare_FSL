import os
import shutil
import random

folders = os.listdir("/opt/huangyanchun/workspace/datasets/imagenet/train_picvec")

random.shuffle(folders)
folders = folders[0:234]
for folder in folders:
    shutil.move("/opt/huangyanchun/workspace/datasets/imagenet/train_picvec/%s" % folder, "/opt/huangyanchun/workspace/datasets/imagenet/val_picvec")
