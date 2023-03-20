import os
import os.path as osp
import torch
import torch.utils.data as Data
from torch.utils.data.sampler import Sampler

import numpy as np
import json


class ClassBalancedSampler(Sampler):
    def __init__(self, data_source: Data.TensorDataset, num_samples=None):
        super(ClassBalancedSampler, self).__init__(data_source)
        self.num_samples = len(data_source) if num_samples is None else num_samples

        targets = [data_source.__getitem__(i)[1].item() for i in range(len(data_source))]
        label_count = [0] * len(np.unique(targets))
        for idx in range(len(data_source)):
            label = targets[idx]
            label_count[label] += 1

        weights_per_cls = 1.0 / np.array(label_count)
        weights = [weights_per_cls[targets[idx]]
                   for idx in range(len(data_source))]
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return iter(torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=True
        ).tolist())

    def __len__(self):
        return self.num_samples



def load_tabular(args):
    """prepare the credit dataset including shuffle and splitting.
        Extraction was done by Barry Becker from the 1994 Census database. Detailed description could be found in http://archive.ics.uci.edu/ml/datasets/Census+Income

        Args:
            args: set containing all program arguments.

        Returns:
            train set, test set, the whole X & y and train_loader.
        """
    data_folder = osp.join(args.data_path, args.dataset)
    X_train = np.load(osp.join(data_folder, "X_train.npy"))
    y_train = np.load(osp.join(data_folder, "y_train.npy"))
    X_test = np.load(osp.join(data_folder, "X_test.npy"))
    y_test = np.load(osp.join(data_folder, "y_test.npy"))
    X_train_sampled = np.load(osp.join(data_folder, "X_train_sampled.npy"))
    y_train_sampled = np.load(osp.join(data_folder, "y_train_sampled.npy"))
    X_test_sampled = np.load(osp.join(data_folder, "X_test_sampled.npy"))
    y_test_sampled = np.load(osp.join(data_folder, "y_test_sampled.npy"))

    if args.dataset in ["census", "commercial"]:
        X_train = torch.from_numpy(X_train).float().to(args.device)
        y_train = torch.from_numpy(y_train).long().to(args.device)
        X_test = torch.from_numpy(X_test).float().to(args.device)
        y_test = torch.from_numpy(y_test).long().to(args.device)
        X_train_sampled = torch.from_numpy(X_train_sampled).float().to(args.device)
        y_train_sampled = torch.from_numpy(y_train_sampled).long().to(args.device)
        X_test_sampled = torch.from_numpy(X_test_sampled).float().to(args.device)
        y_test_sampled = torch.from_numpy(y_test_sampled).long().to(args.device)
    elif args.dataset in ["bike"]:
        X_train = torch.from_numpy(X_train).float().to(args.device)
        y_train = torch.from_numpy(y_train).float().to(args.device)
        X_test = torch.from_numpy(X_test).float().to(args.device)
        y_test = torch.from_numpy(y_test).float().to(args.device)
        X_train_sampled = torch.from_numpy(X_train_sampled).float().to(args.device)
        y_train_sampled = torch.from_numpy(y_train_sampled).float().to(args.device)
        X_test_sampled = torch.from_numpy(X_test_sampled).float().to(args.device)
        y_test_sampled = torch.from_numpy(y_test_sampled).float().to(args.device)
    else:
        raise Exception


    # create dataloader
    train_set = Data.TensorDataset(X_train, y_train)

    if "balance" in vars(args).keys() and args.balance:
        print("################## Use balanced dataset ##################")
        sampler = ClassBalancedSampler(train_set)
        train_loader = Data.DataLoader(train_set, batch_size=args.batch_size, drop_last=True, sampler=sampler)
    else:
        train_loader = Data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    with open(osp.join(data_folder, "info.json"), "r") as f:
        dataset_info = json.load(f)

    return dataset_info,\
           X_train, y_train, X_test, y_test,\
           X_train_sampled, y_train_sampled,\
           X_test_sampled, y_test_sampled,\
           train_loader


def load_tabular_for_adv(args):

    data_folder = osp.join(args.data_path, args.dataset)
    X_train = np.load(osp.join(data_folder, "X_train.npy"))
    y_train = np.load(osp.join(data_folder, "y_train.npy"))
    X_test = np.load(osp.join(data_folder, "X_test.npy"))
    y_test = np.load(osp.join(data_folder, "y_test.npy"))

    if args.dataset in ["census", "commercial"] \
            + [f"commercial_rule{i}_classification" for i in range(1, 20)] \
            + [f"commercial_rule{i}_classification_ind" for i in range(1, 20)] \
            + [f"gaussian_rule{i}_classification_ind" for i in range(1, 20)] \
            + [f"gaussian_8d_rule{i}_classification" for i in range(1, 20)]:
        X_train = torch.from_numpy(X_train).float().to(args.device)
        y_train = torch.from_numpy(y_train).long().to(args.device)
        X_test = torch.from_numpy(X_test).float().to(args.device)
        y_test = torch.from_numpy(y_test).long().to(args.device)
    elif args.dataset in ["bike", "commercial_rule1_regression"]:
        X_train = torch.from_numpy(X_train).float().to(args.device)
        y_train = torch.from_numpy(y_train).float().to(args.device)
        X_test = torch.from_numpy(X_test).float().to(args.device)
        y_test = torch.from_numpy(y_test).float().to(args.device)
    else:
        raise NotImplementedError

    # create dataloader
    train_set = Data.TensorDataset(X_train, y_train)
    if "balance" in vars(args).keys() and args.balance:
        print("################## Use balanced dataset ##################")
        sampler = ClassBalancedSampler(train_set)
        train_loader = Data.DataLoader(train_set, batch_size=args.batch_size, drop_last=True, sampler=sampler)
    else:
        train_loader = Data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_set = Data.TensorDataset(X_test, y_test)
    test_loader = Data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    return X_train, y_train, train_set, test_set,\
           train_loader, test_loader


def get_attributes(data_path, dataset):
    with open(osp.join(data_path, dataset, "info.json"), "r") as f:
        dataset_info = json.load(f)

    return dataset_info["attributes"]


def get_data_mean_std(data_path, dataset):
    with open(osp.join(data_path, dataset, "info.json"), "r") as f:
        dataset_info = json.load(f)
    return np.array(dataset_info['X_mean_original']), np.array(dataset_info['X_std_original'])


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    args = argparse.ArgumentParser().parse_args()
    args.data_path = "/data1/limingjie/data/tabular"
    args.dataset = "commercial"
    args.batch_size = 512
    args.device = 1

    dataset_info, \
    X_train, y_train, X_test, y_test, \
    X_train_sampled, y_train_sampled, \
    X_test_sampled, y_test_sampled, \
    train_loader = load_tabular(args)

    X_train = X_train.cpu().numpy()
    y_train = y_train.cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title("positive samples")
    pos_sample = X_train[y_train == 1]
    pos_sample = [pos_sample[:, i] for i in range(pos_sample.shape[1])]
    print(pos_sample)
    plt.boxplot(pos_sample)
    plt.xticks(list(range(len(dataset_info["attributes"]))), dataset_info["attributes"])
    plt.subplot(122)
    plt.title("negative samples")
    neg_sample = X_train[y_train == 0]
    plt.boxplot([neg_sample[:, i] for i in range(neg_sample.shape[1])])
    plt.xticks(list(range(len(dataset_info["attributes"]))), dataset_info["attributes"])

    plt.tight_layout()
    plt.savefig(f"./{args.dataset}.png", dpi=300)