import os
import os.path as osp
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
import torch.backends.cudnn
from sklearn.preprocessing import LabelEncoder


def makedirs(dir):
    if not osp.exists(dir):
        os.makedirs(dir)


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def set_seed(seed = 0):
    """set the random seed for multiple packages.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





def recognize_features_type(df):
    '''
    classify the fields in a pandas table, according to their data-type
    :param df: the pandas table
    :return: tuple -- (<dict>, <dict>) -- (type->fields, field->type)
    '''
    integer_features = list(df.select_dtypes(include=['int64']).columns)
    double_features = list(df.select_dtypes(include=['float64']).columns)
    string_features = list(df.select_dtypes(include=['object']).columns)
    type_features = {
        'integer': integer_features,
        'double': double_features,
        'string': string_features,
    }
    features_type = dict()
    for col in integer_features:
        features_type[col] = 'integer'
    for col in double_features:
        features_type[col] = 'double'
    for col in string_features:
        features_type[col] = 'string'
        
    return type_features, features_type


def set_discrete_continuous(features, type_features, class_name, discrete=None, continuous=None):
    if discrete is None and continuous is None:
        discrete = type_features['string']
        continuous = type_features['integer'] + type_features['double']
        
    if discrete is None and continuous is not None:
        discrete = [f for f in features if f not in continuous]
        continuous = list(set(continuous + type_features['integer'] + type_features['double']))
        
    if continuous is None and discrete is not None:
        continuous = [f for f in features if f not in discrete and (f in type_features['integer'] or f in type_features['double'])]
        discrete = list(set(discrete + type_features['string']))
    
    discrete = [f for f in discrete if f != class_name] + [class_name]
    continuous = [f for f in continuous if f != class_name]
    return discrete, continuous


def label_encode(df, columns):
    df_le = df.copy(deep=True)  # le: Label_Encode
    label_encoder = dict()
    for col in columns:
        # Encode target labels with value between 0 and n_classes-1
        le = LabelEncoder()
        df_le[col] = le.fit_transform(df_le[col])  # TODO: whether this is reasonable?
        label_encoder[col] = le
    return df_le, label_encoder


def stratified_split(y, train_ratio):
    """split the imbalance dataset into balanced 
    """
    def split_class(y, label, train_ratio):
        indices = np.flatnonzero(y == label)
        n_train = int(y.size*train_ratio/len(np.unique(y)))
        train_index = indices[:n_train]
        test_index = indices[n_train:]
        return (train_index, test_index)
        
    idx = [split_class(y, label, train_ratio) for label in np.unique(y)]
    train_index = np.concatenate([train for train, _ in idx])
    test_index = np.concatenate([test for _, test in idx])
    return train_index, test_index


def get_unmasked_attribute_name(
    mask: torch.BoolTensor,
    attributes: list
) -> list:
    assert len(mask) == len(attributes)
    attributes = np.array(attributes)
    mask = mask.cpu().numpy()
    attributes = list(attributes[mask])
    return attributes