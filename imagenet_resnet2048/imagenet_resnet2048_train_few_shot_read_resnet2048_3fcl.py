# -*- coding: utf-8 -*-
#-------------------------------------
# Project: Learning to Compare: Relation Network for Few-Shot Learning
# Date: 2017.9.21
# Author: Flood Sung
# All Rights Reserved
#-------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator_read_resnet2048 as tg
import os
import math
import argparse
import scipy as sp
import scipy.stats
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 2048)
parser.add_argument("-r","--relation_dim",type = int, default = 400)
parser.add_argument("-w","--class_num",type = int, default = 30)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 20) # 即论文里每个类的sample images的个数
parser.add_argument("-b","--batch_num_per_class",type = int, default = 5)  # 即论文里每个类的query images的个数
parser.add_argument("-e","--episode",type = int, default= 500000)
parser.add_argument("-t","--test_episode", type = int, default = 100)
parser.add_argument("-l","--learning_rate", type = float, default = 1e-5)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-ug","--use_gpu",type=bool, default=True)
parser.add_argument("-u","--hidden_unit",type=int,default=10)   # 没用到
parser.add_argument("-train_f","--train_folder",type=str,default='../../../imagenet/train_picvec')  # 不要以/开头，因为解析时会把第一个/去掉，从而引起报错
parser.add_argument("-test_f","--test_folder",type=str,default='../../../imagenet/val_picvec')  # 不要以/开头，因为解析时会把第一个/去掉，从而引起报错
parser.add_argument("-c_p","--checkpoint_path",type=str,default='./models/imagenet_resnet2048_relation_network_5way_10shot.pkl"')   # 没用到
args = parser.parse_args()

# limit gpu usage
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
USE_GPU = args.use_gpu
HIDDEN_UNIT = args.hidden_unit
TRAIN_FOLDER = args.train_folder
TEST_FOLDER = args.test_folder
CHECKPOINT_PATH = args.checkpoint_path

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,32)
        self.fc3 = nn.Linear(32,1)

    def forward(self,x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


# def accuracy_calc(labels, preds):
#     _, predict_labels = torch.max(preds.data, 1)
#     rewards = [1 if predict_labels[j] == labels[j] else 0 for j in range(labels.shape[0])]
#     total_rewards = np.sum(rewards)
#     accuracy = total_rewards / 1.0 / labels.shape[0]
#
#     return accuracy
#
#
# def loss_calc(labels, preds):
#     mse = nn.MSELoss()
#     one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS * CLASS_NUM, CLASS_NUM).scatter_(1, labels.view(-1, 1), 1))
#     if USE_GPU:
#         mse = mse.cuda(GPU)
#         one_hot_labels = one_hot_labels.cuda(GPU)
#     loss = mse(preds, one_hot_labels)
#
#     return loss


def main():
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    metatrain_folders,metatest_folders = tg.mini_imagenet_folders(TRAIN_FOLDER, TEST_FOLDER)

    # Step 2: init neural networks
    print("init neural networks")
    # feature_encoder = ResNet50(include_top=False, pooling='max', weights=None)
    # feature_encoder.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    relation_network = RelationNetwork(FEATURE_DIM*2,RELATION_DIM)
    relation_network.apply(weights_init)

    if USE_GPU:
        relation_network.cuda(GPU)

    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=20000,gamma=0.5)

    if os.path.exists(CHECKPOINT_PATH):
        if USE_GPU:
            relation_network.load_state_dict(torch.load(CHECKPOINT_PATH))
        else:
            relation_network.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
        print("load relation network success")

    # Step 3: build graph
    print("Training...")

    last_accuracy = 0.0

    for episode in range(EPISODE):

        relation_network_scheduler.step(episode)

        # init dataset
        # sample_dataloader is to obtain previous samples for compare
        # batch_dataloader is to batch samples for training
        task = tg.MiniImagenetTask(metatrain_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
        sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
        batch_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True)

        # sample datas
        samples,sample_labels = sample_dataloader.__iter__().next() #25*3*84*84，sample_labels用不到
        batches,batch_labels = batch_dataloader.__iter__().next()
        if USE_GPU:
            batch_labels = batch_labels.cuda(GPU)

        # calculate features
        sample_features = Variable(samples.view(-1, FEATURE_DIM, 1, 1))
        if USE_GPU:
            sample_features = sample_features.cuda(GPU)
        sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,1,1)
        sample_features = torch.sum(sample_features,1).squeeze(1)
        batch_features = Variable(batches.view(-1, FEATURE_DIM, 1, 1))
        if USE_GPU:
            batch_features = batch_features.cuda(GPU)

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
        batch_features_ext = torch.transpose(batch_features_ext,0,1)
        relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,FEATURE_DIM*2,1,1)
        relation_pairs = relation_pairs.view(relation_pairs.size(0), -1)
        relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

        _, predict_labels = torch.max(relations.data, 1)
        rewards = [1 if predict_labels[j] == batch_labels[j] else 0 for j in range(batch_labels.shape[0])]
        total_rewards = np.sum(rewards)
        accuracy = total_rewards / 1.0 / batch_labels.shape[0]

        mse = nn.MSELoss()
        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS * CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.cpu().view(-1, 1), 1))
        if USE_GPU:
            mse = mse.cuda(GPU)
            one_hot_labels = one_hot_labels.cuda(GPU)
        loss = mse(relations, one_hot_labels)

        # training

        # feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        # torch.nn.utils.clip_grad_norm(feature_encoder.parameters(),0.5)
        # torch.nn.utils.clip_grad_norm(relation_network.parameters(),0.5)

        # feature_encoder_optim.step()
        relation_network_optim.step()

        if (episode+1)%1 == 0:
                print("episode:",episode+1, "train loss:",loss.data, "train accuracy:",accuracy)

        if (episode+1)%2000 == 0:

            # test
            print("Testing...")
            accuracies = []
            test_losses = []
            for i in range(TEST_EPISODE):
                total_rewards = 0
                task = tg.MiniImagenetTask(metatest_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
                sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
                # num_per_class = 5
                test_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=False)

                sample_images,sample_labels = sample_dataloader.__iter__().next()   # sample_labels用不到
                for test_images,test_labels in test_dataloader: # 只会执行循环体一次，因此此处没必要用for循环
                    if USE_GPU:
                        test_labels = test_labels.cuda(GPU)
                    batch_size = test_labels.shape[0]

                    # calculate features
                    sample_features = Variable(sample_images.view(-1, FEATURE_DIM, 1, 1))
                    if USE_GPU:
                        sample_features = sample_features.cuda(GPU)
                    sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,1,1)
                    sample_features = torch.sum(sample_features,1).squeeze(1)
                    test_features = Variable(test_images.view(-1, FEATURE_DIM, 1, 1))
                    if USE_GPU:
                        test_features = test_features.cuda(GPU)

                    # calculate relations
                    # each batch sample link to every samples to calculate relations
                    # to form a 100x128 matrix for relation network
                    sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1,1,1)

                    test_features_ext = test_features.unsqueeze(0).repeat(1*CLASS_NUM,1,1,1,1)
                    test_features_ext = torch.transpose(test_features_ext,0,1)
                    relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,1,1)
                    relation_pairs = relation_pairs.view(relation_pairs.size(0), -1)
                    relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

                    _,predict_labels = torch.max(relations.data,1)
                    rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(batch_size)]
                    total_rewards += np.sum(rewards)

                    mse = nn.MSELoss()
                    one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS * CLASS_NUM, CLASS_NUM).scatter_(1, test_labels.cpu().view(-1, 1), 1))
                    if USE_GPU:
                        mse = mse.cuda(GPU)
                        one_hot_labels = one_hot_labels.cuda(GPU)
                    loss_tmp = float(mse(relations, one_hot_labels).cpu().data.numpy())

                accuracy = total_rewards/1.0/CLASS_NUM/BATCH_NUM_PER_CLASS
                accuracies.append(accuracy)
                test_losses.append(loss_tmp)

            test_accuracy,h = mean_confidence_interval(accuracies)
            test_loss = np.mean(np.array(test_losses))

            print("test accuracy:",test_accuracy, "h:",h, "test loss:",test_loss)

            if test_accuracy > last_accuracy:

                # save networks
                torch.save(relation_network.state_dict(), CHECKPOINT_PATH)

                print("save networks for episode:",episode+1)

                last_accuracy = test_accuracy


if __name__ == '__main__':
    main()
