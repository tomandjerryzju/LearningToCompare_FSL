# -*- coding: utf-8 -*-
#-------------------------------------
# Project: Learning to Compare: Relation Network for Few-Shot Learning
# Date: 2017.9.21
# Author: Flood Sung
# All Rights Reserved
#-------------------------------------

'''
基于原作者的预测脚本imagenet_resnet2048_test_few_shot_predict_keras.py修改而来，分布式预测脚本。
'''

import numpy as np
import os
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing import image
import random
import time
import shutil
from keras.layers import Input, Dense
from keras.models import Model


# limit gpu usage
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))


def RelationNetwork_keras(input_size, hidden_size):
    input = Input(shape=(input_size,))
    x_model = Dense(hidden_size, activation='relu', name='fc1')(input)
    predictions = Dense(1, activation='sigmoid', name='fc2')(x_model)
    model = Model(inputs=input, outputs=predictions)

    return model


def prepare_pic_array(pic_path, img_height=224, img_width=224):
    try:
        img = image.load_img(pic_path, target_size=(img_height, img_width))
        img = img.convert('RGB')
        x = image.img_to_array(img)
        if x.shape != (img_height, img_width, 3):
            print('filepath shape unmatched: %s shape=%s' % (pic_path, x.shape))
            return None
        return x
    except Exception, e:
        print e
        return None


def prepare_feature_per_class(feature_encoder, image_root, support_num_per_class):
    file_list = os.listdir(image_root)
    random.seed(1)
    random.shuffle(file_list)
    pic_arrays = []
    feature = None
    for filename in file_list:
        try:
            file_path = os.path.join(image_root, filename)
            pic_array = prepare_pic_array(file_path)
            if pic_array is None:
                continue
            pic_arrays.append(pic_array)
            if len(pic_arrays) == support_num_per_class:
                try:
                    pic_arrays = np.array(pic_arrays)
                    pic_arrays = preprocess_input(pic_arrays)
                    features = feature_encoder.predict_on_batch(pic_arrays)
                    feature = features.sum(axis=0)
                    break
                except Exception, e:
                    print 'predict error , detail=%s' % e
        except Exception, e:
            print e
            continue
    return feature


def prepare_features_support_set(feature_encoder, support_set, class_num, support_num_per_class):
    image_roots = os.listdir(support_set)
    image_roots_sort = []
    for image_root in image_roots:
        try:
            image_roots_sort.append(int(image_root))
        except Exception, e:
            print e
            continue
    image_roots_sort.sort()
    features = []
    labels = []
    for image_root in image_roots_sort:
        try:
            label = image_root
            image_root = os.path.join(support_set, str(image_root))
            feature = prepare_feature_per_class(feature_encoder, image_root, support_num_per_class)
            if feature is not None:
                features.append(feature)
                labels.append(label)
        except Exception, e:
            print e
            continue
    if len(features) != class_num:
        print "len(features) != class_num"
        exit(1)
    features = np.array(features)
    labels = np.array(labels)
    return features, labels


def batch_predict(feature_encoder, relation_network, support_set, class_num, support_num_per_class, image_root, output_file, batch_size=20):
    file_list = os.listdir(image_root)
    pic_arrays = []
    filename_list = []
    start_time = time.time()
    batch_count = 0
    img_fail = 0
    fout = open(output_file, 'a')
    output_root = os.path.join(os.path.abspath(os.path.dirname(output_file)), 'predict_result')
    support_features, support_labels = prepare_features_support_set(feature_encoder, support_set, class_num, support_num_per_class)
    support_features_ext = np.tile(support_features, (batch_size, 1, 1))
    for filename in file_list:
        try:
            file_path = os.path.join(image_root, filename)
            pic_array = prepare_pic_array(file_path)
            if pic_array is None:
                img_fail = img_fail + 1
                continue
            pic_arrays.append(pic_array)
            filename_list.append(filename)
            if len(pic_arrays) == batch_size:
                try:
                    pic_arrays = np.array(pic_arrays)
                    pic_arrays = preprocess_input(pic_arrays)
                    test_features = feature_encoder.predict_on_batch(pic_arrays)
                    test_features_ext = np.tile(test_features, (class_num, 1, 1))
                    test_features_ext = test_features_ext.transpose((1, 0, 2))
                    relation_pairs = np.concatenate((support_features_ext, test_features_ext), 2)
                    relation_pairs = relation_pairs.reshape(-1, 2048 * 2)
                    relations = relation_network.predict_on_batch(relation_pairs)
                    relations = relations.reshape(-1, class_num)
                    for i in range(len(relations)):
                        predict_label = np.argmax(relations[i])
                        predict_score = np.max(relations[i])
                        line = [filename_list[i], str(predict_label), str(predict_score)]
                        line = '\t'.join(line) + '\n'
                        fout.write(line)
                        if is_cp and predict_score > 0.4:
                            if not os.path.exists(output_root):
                                os.mkdir(output_root)
                            tmp_fname = filename_list[i]
                            src_path = os.path.join(image_root, tmp_fname)
                            new_fname = '{}_{}_{}.jpg'.format(tmp_fname[:-4], predict_score, predict_label)
                            print 'new filename: %s' % new_fname
                            tar_path = os.path.join(output_root, new_fname)
                            shutil.copyfile(src=src_path, dst=tar_path)
                    fout.flush()
                except Exception, e:
                    print 'predict error in batch, detail=%s' % e
                print 'batch %d done, time %f, %d' % (batch_count, time.time() - start_time, img_fail)
                batch_count += 1
                pic_arrays = []
                filename_list = []
                # break # debug
        except Exception, e:
            print e
            continue
    if len(pic_arrays) > 0:
        try:
            support_features_ext = np.tile(support_features, (len(pic_arrays), 1, 1))
            pic_arrays = np.array(pic_arrays)
            pic_arrays = preprocess_input(pic_arrays)
            test_features = feature_encoder.predict_on_batch(pic_arrays)
            test_features_ext = np.tile(test_features, (class_num, 1, 1))
            test_features_ext = test_features_ext.transpose((1, 0, 2))
            relation_pairs = np.concatenate((support_features_ext, test_features_ext), 2)
            relation_pairs = relation_pairs.reshape(-1, 2048 * 2)
            relations = relation_network.predict_on_batch(relation_pairs)
            relations = relations.reshape(-1, class_num)
            for i in range(len(relations)):
                predict_label = np.argmax(relations[i])
                predict_score = np.max(relations[i])
                line = [filename_list[i], str(predict_label), str(predict_score)]
                line = '\t'.join(line) + '\n'
                fout.write(line)
                if is_cp and predict_score > 0.4:
                    if not os.path.exists(output_root):
                        os.mkdir(output_root)
                    tmp_fname = filename_list[i]
                    src_path = os.path.join(image_root, tmp_fname)
                    new_fname = '{}_{}_{}.jpg'.format(tmp_fname[:-4], predict_score, predict_label)
                    print 'new filename: %s' % new_fname
                    tar_path = os.path.join(output_root, new_fname)
                    shutil.copyfile(src=src_path, dst=tar_path)
            fout.flush()
        except Exception, e:
            print 'predict error in batch, detail=%s' % e
    print 'done'


def main():
    print("init neural networks")
    feature_encoder = ResNet50(include_top=False, pooling='max', weights=None)
    feature_encoder.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    support_set = "/Users/hyc/workspace/LearningToCompare_FSL/datas/imagenet_resnet2048/test_v3/"
    image_root = "/Users/hyc/workspace/LearningToCompare_FSL/datas/imagenet_resnet2048/test_v3/0"
    output_file = "result.txt"
    relation_network = RelationNetwork_keras(2048 * 2, 400)
    checkpoint_path = "/Users/hyc/workspace/LearningToCompare_FSL/imagenet_resnet2048/models/relation_network_keras.h5"
    relation_network.load_weights(checkpoint_path)
    print("load relation network success")
    batch_predict(feature_encoder, relation_network, support_set, 6, 10, image_root, output_file, batch_size=20)


if __name__ == '__main__':
    is_cp = 1
    main()
