# -*- coding: utf-8 -*-
import codecs

import cPickle
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
import time
from keras.applications.imagenet_utils import preprocess_input
import os
import sys
import keras
import random
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from PIL import Image

# limit gpu usage
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.visible_device_list = "0"
# set_session(tf.Session(config=config))
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

PARENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PARENT_PATH)
PROJECT_PATH = PARENT_PATH
print('Relate file: %s, \tPROJECT path = %s\nPARENT PATH = %s' % (__file__, PROJECT_PATH, PARENT_PATH))
sys.path.append(PROJECT_PATH)


def prepare_pic_array(pic_path, img_height, img_width):
    try:
        # img = image.load_img(pic_path, target_size=(img_height, img_width))
        img = Image.open(pic_path)
        img = img.resize((224, 224))
        img = img.convert('RGB')
        x = image.img_to_array(img)
        if x.shape != (img_height, img_width, 3):
            print('filepath shape unmatched: %s shape=%s' % (pic_path, x.shape))
            return None
        return x
    except Exception, e:
        print e
        return None


def calc_resnet2048(pic_path, img_height, img_width, model):
    x = prepare_pic_array(pic_path, img_height, img_width)
    if x is None:
        return None
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    pic_vec = model.predict_on_batch(x)[0]
    return pic_vec


def save_top_feature(file_root, model, output_root, img_height, img_width):
    file_list = os.listdir(file_root)
    start_time = time.time()
    img_fail = 0
    cnt = 0
    for filename in file_list:
        try:
            file_path = os.path.join(file_root, filename)
            pic_vec = calc_resnet2048(file_path, img_height, img_width, model)
            if pic_vec is None or len(pic_vec) != 2048:
                img_fail = img_fail + 1
                continue
            savefile = os.path.join(output_root, "%s.npy" % '.'.join(filename.split('.')[0:-1]))
            np.save(savefile, pic_vec)
            cnt = cnt + 1
            print '%d done, time %f, %d fail' % (cnt, time.time() - start_time, img_fail)
        except IOError, ioe:
            print ioe
            continue
        except Exception, e:
            print e
            continue
    print 'done!'


def deal_with_dataset(dataset, model):
    folders = os.listdir(dataset)
    for folder in folders:
        if folder.startswith('.'):
            continue
        file_root = os.path.join(dataset, folder)
        output_path = os.path.join(os.path.dirname(dataset), dataset.split('/')[-1] + '_picvec')
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        output_root = os.path.join(output_path, folder)
        if not os.path.exists(output_root):
            os.mkdir(output_root)
        save_top_feature(file_root, model, output_root, 224, 224)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'usage : python resnet50_fcl_train.py dataset'
        exit(1)

    dataset = sys.argv[1]

    cnn_model = ResNet50(include_top=False, pooling='max', weights=None)
    cnn_model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    deal_with_dataset(dataset, cnn_model)
