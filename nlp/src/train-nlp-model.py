import os
import os.path as osp
import argparse
import torch
import torch.nn as nn
from tools.train import train_model
from model import load_nlp_model
from set_exp import makedirs_for_train_model
from dataset import load_nlp_dataset

parser = argparse.ArgumentParser(description="Train NLP models")
parser.add_argument('--gpu_id', default=2, type=int, help="set the gpu id, use -1 to indicate cpu.")
parser.add_argument("--dataset", default="CoLA", type=str, help="set the dataset used: SST-2, CoLA")
parser.add_argument("--arch", default="cnn", type=str, help="the network architecture: cnn, lstm2_uni")
parser.add_argument('--data_path', default='/data1/limingjie/data/NLP', type=str, help="root of datasets.")
parser.add_argument("--model_path", default="../saved-models-nlp", type=str, help='where to save the model.')

parser.add_argument("--seed", default=0, type=int, help="set the seed used for training model.")
parser.add_argument('--batch_size', default=64, type=int, help="set the batch size for training.")
parser.add_argument('--train_lr', default=0.001, type=float, help="set the learning rate for training.")
parser.add_argument("--logspace", default=1, type=int, help='the decay of learning rate.')
parser.add_argument("--epoch", default=100, type=int, help='the number of iterations for training model.')

args = parser.parse_args()
makedirs_for_train_model(args)

# ===============================================
#   prepare the dataset (for train & eval)
# ===============================================
print()
print('-' * 40)
print("-----------preparing dataset-----------")
print('-' * 40)
train_set, test_set, train_iterator, test_iterator, TEXT, LABEL = load_nlp_dataset(args)

# ===============================================
#   initialize the model
# ===============================================
print()
print('-' * 40)
print("-------------load the model-------------")
print('-' * 40)
net = load_nlp_model(args, TEXT, LABEL, model_path=None)

# ===============================================
#   train the model
# ===============================================
print()
print('-' * 40)
print("-----------start training-----------")
print('-' * 40)
net = train_model(args, net, train_iterator, test_iterator)
