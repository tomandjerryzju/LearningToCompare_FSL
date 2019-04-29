import os
import shutil
import random

path = "/opt/huangyanchun/workspace/datasets/few_shot_learning/LearningToCompare_FSL/datas/imagenet_resnet2048/all_extentpic_ugcpic_checked/fsl_normal_dataset_val_picvec"
folders = os.listdir(path)

# random.shuffle(folders)
# folders = folders[0:253]
for folder in folders:
    count = os.listdir(os.path.join(path, folder))
    if len(count) < 50:
        shutil.rmtree(os.path.join(path, folder))
        print folder

    # shutil.move("/opt/huangyanchun/workspace/datasets/few_shot_learning/LearningToCompare_FSL/datas/imagenet_resnet2048/all_extentpic_ugcpic_checked/fsl_normal_dataset_train_picvec/%s" % folder, "/opt/huangyanchun/workspace/datasets/few_shot_learning/LearningToCompare_FSL/datas/imagenet_resnet2048/all_extentpic_ugcpic_checked/fsl_normal_dataset_val_picvec")
