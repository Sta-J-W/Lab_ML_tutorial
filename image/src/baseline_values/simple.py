import numpy as np
import torch


def mean_baseline(X_train):
    '''
    Use **mean** value in each attribute dimension as the reference value
    :param X_train: data sampled from p_data, torch.Size([n_samples, n_attributes])
    :return: baseline value in each dimension, torch.Size([1, n_attributes,])
    '''
    baseline = torch.mean(X_train, dim=0, keepdim=True)
    return baseline


def zero_baseline(X_train):
    '''
    Use **zero** value in each attribute dimension as the reference value
    :param X_train: data sampled from p_data, torch.Size([n_samples, n_attributes])
    :return: baseline value in each dimension, torch.Size([1, n_attributes,])
    '''
    baseline = torch.zeros_like(X_train)
    return baseline


def damp_baseline(X_train, lamb_damp):
    assert X_train.shape[0] == 1
    assert 0 <= lamb_damp <= 1
    return X_train * lamb_damp


def get_baseline_value(X_train, baseline_config):
    if baseline_config == "zero":
        return zero_baseline(X_train)
    elif baseline_config == "mean":
        return mean_baseline(X_train)
    elif baseline_config.startswith("damp_"):
        lamb_damp = baseline_config.split("_")[-1]
        lamb_damp = float(lamb_damp)
        return damp_baseline(X_train, lamb_damp=lamb_damp)
    else:
        raise NotImplementedError(f"Unknown baseline configuration: {baseline_config}")