import os
import os.path as osp
import logging
import argparse
import torch
import torch.nn as nn
import model
from tools.train import train_model
import dataset
from set_exp import makedirs_for_train_model
import numpy as np
from tqdm import tqdm


print("-------Parsing program arguments--------")
parser = argparse.ArgumentParser(description="train model code")
## the basic setting of exp
parser.add_argument('--device', default=2, type=int,
                    help="set the device.")
parser.add_argument("--dataset", default="commercial", type=str,
                    help="set the dataset used: commercial, census, bike")
parser.add_argument("--arch", default="mlp5", type=str,
                    help="the network architecture: mlp5, resmlp5, mlp2_logistic")
# set the path for data
parser.add_argument('--data_path', default='/data1/limingjie/data/tabular', type=str,
                    help="path for dataset.")
# set the (pre-trained) model path.
parser.add_argument("--model_path", default="../saved-models", type=str,
                    help='the path of pretrained model.')

## the setting for the pre-trained model
# set the model seed
parser.add_argument("--model_seed", default=0, type=int,
                    help="set the seed used for training model.")
# set the batch size for training
parser.add_argument('--batch_size', default=512, type=int,
                    help="set the batch size for training.")
# set the learning rate for training
parser.add_argument('--train_lr', default=0.01, type=float,
                    help="set the learning rate for training.")
# set the decay of learning rate
parser.add_argument("--logspace", default=1, type=int,
                    help='the decay of learning rate. if set as 1, then lr will '
                         'decay exponentially for 10x over the training process.')
# set the number of epochs for training model.
parser.add_argument("--epoch", default=300, type=int,
                    help='the number of iterations for training model.')

# optional: whether to rebalance the data
parser.add_argument("--balance", action="store_true",
                    help="Set this flag if you need data re-sampling for "
                         "class-imbalanced datasets (e.g. census).")

args = parser.parse_args()
makedirs_for_train_model(args)


# ===============================================
#   prepare the dataset (for train & eval)
# ===============================================
print("-----------preparing dataset-----------")
print("dataset - {}".format(args.dataset))
dataset_info, X_train, y_train, X_test, y_test, \
X_train_sampled, y_train_sampled, X_test_sampled, y_test_sampled, \
train_loader = dataset.load_tabular(args)

if "logistic" in args.arch:
    task = {
        "census": "logistic_regression",
        "commercial": "logistic_regression",
    }[args.dataset]
else:
    task = {
        "census": "classification",
        "commercial": "classification",
        "bike": "regression",
    }[args.dataset]


# ===============================================
#   train the model
# ===============================================
print("------------preparing model------------")
model = train_model(args, X_train, y_train, X_test, y_test, train_loader, task)
model = model.to(args.device)
model.eval()

