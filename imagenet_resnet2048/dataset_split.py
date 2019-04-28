import os
import shutil
import random

folders = os.listdir("/opt/huangyanchun/workspace/datasets/few_shot_learning/LearningToCompare_FSL/datas/imagenet_resnet2048/all_extentpic_ugcpic_checked/fsl_normal_dataset_train_picvec")
# folders_filter = []
# for folder in folders:
#     if folder.startswith('n'):
#         folders_filter.append(folder)

random.shuffle(folders)
folders_filter = folders[0:234]
for folder in folders_filter:
    shutil.move("/opt/huangyanchun/workspace/datasets/few_shot_learning/LearningToCompare_FSL/datas/imagenet_resnet2048/all_extentpic_ugcpic_checked/fsl_normal_dataset_train_picvec/%s" % folder, "/opt/huangyanchun/workspace/datasets/few_shot_learning/LearningToCompare_FSL/datas/imagenet_resnet2048/all_extentpic_ugcpic_checked/fsl_normal_dataset_val_picvec")
