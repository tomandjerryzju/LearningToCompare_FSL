# coding=utf-8

"""
Created by jayveehe on 2018/1/26.
http://code.dianpingoa.com/hejiawei03
"""

import cPickle

from keras import optimizers, Input
from keras.engine import Model
import tensorflow as tf
from keras.layers import Dense, Lambda, Dropout
import keras


class UserModelFactory:
    def __init__(self):
        pass

    is_pairwise = False

    @staticmethod
    def create_model():
        """
        define your model structure here
        :return: a keras Model with 'predict_on_batch()'
        """

        # an example for pairwise dnn
        def build_basic_dnn(res_input, is_train=True):
            x_model = Dense(64, activation='relu', name='dense_1')(res_input)
            x_model = Dropout(0.5, name='dropout_1')(x_model)
            predictions = Dense(2, activation='softmax', name='dense_2')(x_model)
            new_model = Model(inputs=res_input, outputs=predictions)
            #     pairwise_model = prediction
            return new_model

        # resnet_model = keras.applications.ResNet50(include_top=False, weights=None, pooling='max')
        # dnn_model = keras.models.load_model(dnn_h5_path, custom_objects={'sigmoid_acc': sigmoid_acc})
        # dnn_model.load_weights(dnn_h5_path)
        resnet_output = Input(batch_shape=(None, 2048), name='resnet_notop_output')
        dnn_model = build_basic_dnn(resnet_output, is_train=False)
        # dnn_score_model = dnn_model.layers[2]
        # predictions = dnn_score_model(resnet_model.output)
        # combined_model = Model(inputs=resnet_model.input, outputs=predictions)
        return dnn_model

    @staticmethod
    def load_model_weights(pkl_path, model):
        print 'loading model from %s' % pkl_path
        wfin = tf.gfile.GFile(pkl_path, 'rb')
        wlist = cPickle.load(wfin)
        model.set_weights(wlist)
        wfin.close()
        return model
