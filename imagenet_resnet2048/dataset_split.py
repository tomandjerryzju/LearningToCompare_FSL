import os
import shutil
import random

folders = os.listdir("/opt/huangyanchun/workspace/datasets/imagenet/train_extent_picvec")
# folders_filter = []
# for folder in folders:
#     if folder.startswith('n'):
#         folders_filter.append(folder)

random.shuffle(folders)
folders = folders[0:34]
for folder in folders:
    shutil.move("/opt/huangyanchun/workspace/datasets/imagenet/train_extent_picvec/%s" % folder, "/opt/huangyanchun/workspace/datasets/imagenet/val_extent_picvec")
