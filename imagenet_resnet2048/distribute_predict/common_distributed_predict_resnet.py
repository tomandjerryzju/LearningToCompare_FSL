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
from keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf
from keras.preprocessing import image
import random
import time
from keras.layers import Input, Dense
from keras.models import Model
import cPickle
import datetime
import urllib3
import requests
import keras
import gevent
import StringIO
from PIL import Image
import re

urllib3.disable_warnings()
requests.adapters.DEFAULT_RETRIES = 5
resnet_model = keras.applications.ResNet50(include_top=False, weights=None, pooling='max')

def RelationNetwork_keras(input_size, hidden_size):
    input = Input(shape=(input_size,))
    x_model = Dense(hidden_size, activation='relu', name='fc1')(input)
    predictions = Dense(1, activation='sigmoid', name='fc2')(x_model)
    model = Model(inputs=input, outputs=predictions)

    return model


def prepare_pic_array(pic_path, img_height=224, img_width=224):
    try:
        img_bytes = StringIO.StringIO(tf.gfile.GFile(pic_path,'rb').read())
        img = Image.open(img_bytes)
        # img = image.load_img(pic_path, target_size=(img_height, img_width))
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


def prepare_feature_per_class(feature_encoder, image_root, support_num_per_class):
    # file_list = os.listdir(image_root)
    file_list = tf.gfile.ListDirectory(image_root)
    random.seed(1)
    random.shuffle(file_list)
    pic_arrays = []
    feature = None
    for filename in file_list:
        try:
            if filename.startswith('.'):
                continue
            # file_path = os.path.join(image_root, filename)
            file_path = image_root + '/' + filename
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
    # image_roots = os.listdir(support_set)
    image_roots = tf.gfile.ListDirectory(support_set)
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
            # image_root = os.path.join(support_set, str(image_root))
            image_root = support_set + '/' + str(image_root)
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


def get_resnet_from_img(img_arr, resnet_model):
    vec = resnet_model.predict_on_batch(preprocess_input(np.array([img_arr])))
    return vec[0]


def convert_url(url):
    pat = re.compile("p[0-9].meituan.net")
    url = re.sub(pat, "download-image.sankuai.com", url)
    pat = re.compile("img.meituan.net")
    url = re.sub(pat, "download-image.sankuai.com", url)
    pat = re.compile("p.vip.sankuai.com")
    url = re.sub(pat, "download-image.sankuai.com", url)
    pat = re.compile("mtmos.com")
    url = re.sub(pat, "mss.vip.sankuai.com", url)
    pat = re.compile("mss.sankuai.com")
    url = re.sub(pat, "mss.vip.sankuai.com", url)
    pat = re.compile("s3plus.sankuai.com")
    url = re.sub(pat, "s3plus.vip.sankuai.com", url)
    pat = re.compile("mss-shon.sankuai.com")
    url = re.sub(pat, "mss-shon.vip.sankuai.com", url)
    return url


def prepare_img_with_resnet(picid, picurl, picvec, extra_info=None, retry_times=3, target_h=224, target_w=224,
                            resnet_model_inst=resnet_model):
    if picvec:
        return picvec, picid, picurl, 1.5, extra_info, True, 'ok'
    else:
        s = requests.session()
        s.keep_alive = False
        for i in xrange(retry_times):
            try:
                img_bytes = s.get(convert_url(picurl), verify=False, timeout=10.0,
                                  headers={'Connection': 'close'}).content
                break
            except requests.Timeout, te:
                print 'timeout in download %s, details=%s' % (picurl, te)
                if i < retry_times:
                    time.sleep(2)
                    continue
                else:
                    raise te
            except Exception, e:
                print 'error in download %s, details=%s' % (picurl, e)
                if i < retry_times:
                    time.sleep(2)
                    continue
                else:
                    raise e
        try:
            img_ = Image.open(StringIO.StringIO(img_bytes))
            width, height = img_.size
            ratio = width / (height + 0.0)
            img_resize = img_.resize((224, 224))
            x = image.img_to_array(img_resize.convert('RGB'))
            if x.shape != (target_h, target_w, 3):
                x = x[:target_h, :target_w, :3]
            if x.size != target_w * target_h * 3:
                print 'img %s size error: %s,shape: %s, expected %s' % (
                    picurl, x.size, x.shape, target_w * target_h * 3)
                vec = get_resnet_from_img(np.zeros((target_h, target_w, 3)), resnet_model_inst)
                return vec, picid, picurl, 0, extra_info, False, 'grey pic'  # greyscale img
            vec = get_resnet_from_img(x, resnet_model_inst)
            return vec, picid, picurl, ratio, extra_info, True, 'ok'
        except Exception, e:
            print 'error in process img: %s, details=%s' % (picurl, e)
            vec = get_resnet_from_img(np.zeros((target_h, target_w, 3)), resnet_model_inst)
            return vec, picid, picurl, 0, extra_info, False, str(e)


def prepare_gevent_batch(url_gen, batch_size=100):
    from gevent import monkey
    monkey.patch_all()
    multi_results = []
    count = 1
    non_vec_url_count = 0
    print '%s %s done' % (datetime.datetime.now(), count)
    for info_item in url_gen:
        picid = info_item[FLAGS.picid_index]
        picurl = info_item[FLAGS.picurl_index]
        try:
            picvec = [float(a) for a in info_item[FLAGS.picvec_index].strip().split('\x02')]
        except Exception, e:
            non_vec_url_count += 1
            picvec = None
        picvec = picvec if picvec and len(picvec) == 2048 else None
        extra_info = []
        for i in xrange(len(info_item)):
            if i != FLAGS.picvec_index:
                extra_info.append(info_item[i])
        event = gevent.spawn(prepare_img_with_resnet, picid, picurl, picvec, extra_info=extra_info)
        multi_results.append(event)
        count += 1
        if count % batch_size == 0:
            res = gevent.joinall(multi_results)
            tmp_batch = []
            tmp_info = []
            for i in res:
                tmp_res = i.value
                if tmp_res:
                    img_arr, picid, img_url, ratio, extra_info, is_ok, msg = tmp_res
                    tmp_info.append((picid, img_url, ratio, extra_info, is_ok, msg))
                    tmp_batch.append(img_arr)
            print '%s %s done, success %s, non vec count=%s' % (
                datetime.datetime.now(), count, len(tmp_info), non_vec_url_count)
            if len(tmp_batch):
                tmp_batch = np.array(tmp_batch)
                non_vec_url_count = 0
                yield tmp_batch, tmp_info
            multi_results = []
    # final process
    if len(multi_results) > 0:
        res = gevent.joinall(multi_results)

        tmp_batch = []
        tmp_info = []
        for i in res:
            tmp_res = i.value
            if tmp_res:
                img_arr, picid, img_url, ratio, extra_info, is_ok, msg = tmp_res
            tmp_info.append((picid, img_url, ratio, extra_info, is_ok, msg))
            tmp_batch.append(img_arr)
        print '%s %s done, success %s' % (datetime.datetime.now(), count, len(tmp_info))
        if len(tmp_batch):
            tmp_batch = np.array(tmp_batch)
            yield tmp_batch, tmp_info


def batch_predict(feature_encoder, relation_network, support_set, class_num, support_num_per_class, url_generator, output_file, batch_size=20):
    if tf.gfile.Exists(output_file):
        file_mode = 'a'
    else:
        file_mode = 'w'
    support_features, support_labels = prepare_features_support_set(feature_encoder, support_set, class_num, support_num_per_class)
    # print support_features
    with tf.gfile.GFile(output_file, file_mode) as fout:
        for tmp_batch, tmp_info in prepare_gevent_batch(url_generator, batch_size):
            try:
                support_features_ext = np.tile(support_features, (len(tmp_batch), 1, 1))
                test_features = tmp_batch
                test_features_ext = np.tile(test_features, (class_num, 1, 1))
                test_features_ext = test_features_ext.transpose((1, 0, 2))
                relation_pairs = np.concatenate((support_features_ext, test_features_ext), 2)
                relation_pairs = relation_pairs.reshape(-1, 2048 * 2)
                relations = relation_network.predict_on_batch(relation_pairs)
                relations = relations.reshape(-1, class_num)

                for index in range(len(relations)):
                    predict_label = np.argmax(relations[index])
                    predict_score = np.max(relations[index])
                    picid, img_url, ratio, extra_info, is_ok, msg = tmp_info[index]
                    if is_ok:
                        pass
                    else:
                        predict_label = -1
                        predict_score = -1
                    tmp_data = extra_info + [str(predict_label), str(predict_score), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                    out_line = '\t'.join(tmp_data) + '\n'
                    fout.write(out_line)
                tf.logging.info('predict batch len=%s done, flushing' % len(relations))
                fout.flush()
            except Exception, e:
                print('error: %s' % e)
    print 'done'


def url_generator_with_list(input_path_list):
    for input_path in input_path_list:
        for a in tf.gfile.GFile(input_path):
            ul = a.strip().split('\t')
            yield ul


if __name__ == '__main__':
    import sys
    import math

    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string("pkl_path", "", "pickle model path")
    flags.DEFINE_string("url_root", "url.out", "url root")
    flags.DEFINE_string("output_root", "", "result output root")
    flags.DEFINE_integer("batch_size", 20, "batch size")
    flags.DEFINE_integer("proc_count", 10, "multi-process core count")
    flags.DEFINE_integer("task_index", 0, "Index of task within the job")
    flags.DEFINE_integer("picid_index", 0, "Index of picid in input data")
    flags.DEFINE_integer("picurl_index", 1, "Index of picurl in input data")
    flags.DEFINE_integer("picvec_index", 2, "Index of picvec in input data")
    flags.DEFINE_string("model_define_name",
                        'build_naive_dnn_sub_pairwise_model.py',
                        "model define file name with .py")
    flags.DEFINE_integer("workers", 20, "work node num")
    flags.DEFINE_string("log_dir", "", "log dir")
    flags.DEFINE_string("worker_hosts", '20', "work node num")
    flags.DEFINE_string("ps_hosts", '20', "work node num")
    flags.DEFINE_string("job_name", "worker", "job_name")
    flags.DEFINE_string("undefork", "", "list of undefined args")
    # flags.DEFINE_string("support_set", "", "support set")
    # FLAGS(sys.argv)
    pkl_path = FLAGS.pkl_path
    url_root = FLAGS.url_root
    output_root = FLAGS.output_root
    workers = FLAGS.workers
    # support_set = FLAGS.support_set

    yesterday = datetime.datetime.strftime(datetime.datetime.now() - datetime.timedelta(1), '%Y-%m-%d')

    print 'url root=%s' % url_root
    url_file_list = tf.gfile.ListDirectory(url_root)
    print 'total file list: ', len(url_file_list)
    total_file_num = len(url_file_list)
    file_per_worker = int(math.floor(total_file_num / (workers * 1.0)))
    print 'file per worker', file_per_worker
    cur_file_names = url_file_list[FLAGS.task_index:total_file_num:workers]
    print 'assigned file len=%s, first filename=%s, last filename=%s,\nassigned files: %s' % (
        len(cur_file_names), cur_file_names[0], cur_file_names[-1], cur_file_names)
    url_path_list = [url_root + '/' + fname for fname in cur_file_names]

    res_pkl_path = 'viewfs://hadoop-meituan/ghnn01/user/hadoop-dpsr/hejiawei03/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.pkl'
    wfin = tf.gfile.GFile(res_pkl_path, 'rb')
    wlist = cPickle.load(wfin)
    resnet_model.set_weights(wlist)
    wfin.close()
    print 'resnet init done'
    url_gen = url_generator_with_list(url_path_list)
    output_path = output_root + '/result_%s_%s.csv' % (yesterday, FLAGS.task_index)

    model = RelationNetwork_keras(2048 * 2, 400)
    wfin = tf.gfile.GFile(pkl_path, 'rb')
    wlist = cPickle.load(wfin)
    model.set_weights(wlist)
    wfin.close()
    print 'RelationNetwork_keras init done'

    # params need to be set
    class_num = 30
    support_num_per_class = 20
    support_set = "viewfs://hadoop-meituan/ghnn01/user/hadoop-dpsr/huangyanchun/fsl/support_set/support_set_tagname"
    batch_predict(resnet_model, model, support_set, class_num, support_num_per_class, url_gen, output_path, batch_size=200)
    print 'batch size: %s' % FLAGS.batch_size
    print 'output path: %s' % output_path
