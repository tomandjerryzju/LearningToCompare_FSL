# -*- coding: utf-8 -*-
'''
author: huangyanchun
date: 2018/07/31
'''

import sys
import os
import shutil
import random
import re
import hashlib
from mv_file import mvFile
from cp_file import cpFile

def mvFilesForExtentpic(inputParentPath, outputParentPath):
    '''
    基于merge_ugc_and_extentpic.py脚本处理好的数据集，将每个类别下的聚类类别的数据都单独作为不同的类别
    '''
    firstFolders = os.listdir(inputParentPath)
    for firstFolder in firstFolders:
        firstFolderPath = os.path.join(inputParentPath, firstFolder)
        if not os.path.isdir(firstFolderPath):
            print "%s is not a directory" % firstFolder
            continue
        secondFolders = os.listdir(firstFolderPath)
        if len(secondFolders) > 20:
            print "There are no folders in %s" % firstFolder
            continue
        for secondFolder in secondFolders:
            SecondFolderPath = os.path.join(firstFolderPath, secondFolder)
            if not os.path.isdir(SecondFolderPath):
                print "%s is not a directory" % secondFolder
                continue
            output = os.path.join(outputParentPath, firstFolder + '_' + secondFolder)
            shutil.copytree(SecondFolderPath, output)
            # oldname = os.path.join(outputParentPath, secondFolder)
            # newname = os.path.join(outputParentPath, firstFolder + '_' + secondFolder)
            # os.rename(oldname, newname)
    print "done"


if __name__ == "__main__":
    inputParentPath = sys.argv[1]
    outputParentPath = sys.argv[2]

    mvFilesForExtentpic(inputParentPath, outputParentPath)