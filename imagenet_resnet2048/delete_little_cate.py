import os
import shutil
import random

folders = os.listdir("/opt/huangyanchun/workspace/datasets/few_shot_learning/LearningToCompare_FSL/datas/imagenet_resnet2048/all_extentpic_ugcpic_checked/fsl_normal_dataset_train_picvec")

# random.shuffle(folders)
# folders = folders[0:253]
for folder in folders:
    count = os.listdir("/opt/huangyanchun/workspace/datasets/few_shot_learning/LearningToCompare_FSL/datas/imagenet_resnet2048/all_extentpic_ugcpic_checked/fsl_normal_dataset_train_picvec/%s" % folder)
    if count < 50:
        shutil.rmtree("/opt/huangyanchun/workspace/datasets/few_shot_learning/LearningToCompare_FSL/datas/imagenet_resnet2048/all_extentpic_ugcpic_checked/fsl_normal_dataset_train_picvec/%s" % folder)

    # shutil.move("/opt/huangyanchun/workspace/datasets/few_shot_learning/LearningToCompare_FSL/datas/imagenet_resnet2048/all_extentpic_ugcpic_checked/fsl_normal_dataset_train_picvec/%s" % folder, "/opt/huangyanchun/workspace/datasets/few_shot_learning/LearningToCompare_FSL/datas/imagenet_resnet2048/all_extentpic_ugcpic_checked/fsl_normal_dataset_val_picvec")
