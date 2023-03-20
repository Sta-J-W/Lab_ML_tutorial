import numpy as np
import torch


def mean_baseline(X_train):
    '''
    Use **mean** value in each attribute dimension as the reference value
    :param X_train: data sampled from p_data, torch.Size([n_samples, n_attributes])
    :return: baseline value in each dimension, torch.Size([n_attributes,])
    '''
    baseline = torch.mean(X_train, dim=0)
    return baseline


def zero_baseline(X_train):
    '''
    Use **zero** value in each attribute dimension as the reference value
    :param X_train: data sampled from p_data, torch.Size([n_samples, n_attributes])
    :return: baseline value in each dimension, torch.Size([n_attributes,])
    '''
    baseline = torch.zeros(X_train.shape[1])
    return baseline