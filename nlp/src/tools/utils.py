import os
import os.path as osp
import numpy as np
import pickle
import torch
import random
import torch.backends.cudnn


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


def get_all_emb(model, TEXT, device):
    vocab_size = len(TEXT.vocab)
    tensor = torch.LongTensor(list(range(vocab_size))).to(device)
    emb = model.get_emb(tensor)
    return emb

