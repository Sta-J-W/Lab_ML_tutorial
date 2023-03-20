import numpy as np
import torch
import torch.nn as nn


def get_iou(i_1, i_2, eps=1e-7):
    """
    This metric is defined as follows,
        IoU(I1, I2) := \sum_{S\subseteq N} max(|I1(S), I2(S)|) / \sum_{S\subseteq N} min(|I1(S), I2(S)|)
    :param i_1:
    :param i_2:
    :param eps:
    :return:
    """
    i_1 = torch.abs(i_1)
    i_2 = torch.abs(i_2)
    iou = torch.minimum(i_1, i_2).sum() / (torch.maximum(i_1, i_2).sum() + eps)
    return iou


def get_pos_neg_iou(i_1, i_2, eps=1e-7):
    """
    This metric evaluated the IoU w.r.t. the positive/negative parts respectively,
        IoU'(I1, I2) := r * IoU(I1^+, I2^+) + (1-r) * IoU(I1^-, I2^-), where IoU is defined above.
        r = |max(I1^+, I2^+)|_1 / (|max(I1^+, I2^+)|_1 + |max(I1^-, I2^-)|_1)
    :param i_1:
    :param i_2:
    :param eps:
    :return:
    """
    i_1_pos = torch.clamp(i_1, min=0)
    i_1_neg = - torch.clamp(i_1, max=0)
    i_2_pos = torch.clamp(i_2, min=0)
    i_2_neg = - torch.clamp(i_2, max=0)
    iou_pos = get_iou(i_1_pos, i_2_pos, eps=eps)
    iou_neg = get_iou(i_1_neg, i_2_neg, eps=eps)
    r = torch.maximum(i_1_pos, i_2_pos).sum() / (torch.maximum(i_1_pos, i_2_pos).sum()
                                                 + torch.maximum(i_1_neg, i_2_neg).sum() + eps)
    return iou_pos * r + iou_neg * (1 - r)


def get_overlap(i_1, i_2, eps=1e-7, normalize=False):
    """
    The overlap metric is defined as follows, which is similar to IoU,
        overlap(I1, I2) := numerator / denorminator
          -> numerator := \sum_S min(|I1^+(S)|, |I2^+(S)|) + min(|I1^-(S)|, |I2^-(S)|)
          -> denominator := \sum_S max(|I1^+(S)|, |I2^+(S)|) + max(|I1^-(S)|, |I2^-(S)|)
    :param i_1:
    :param i_2:
    :param eps:
    :return:
    """
    if normalize:
        i_1 = i_1 / (torch.sum(torch.abs(i_1)) + eps)
        i_2 = i_2 / (torch.sum(torch.abs(i_2)) + eps)
    i_1_pos = torch.clamp(i_1, min=0)
    i_1_neg = - torch.clamp(i_1, max=0)
    i_2_pos = torch.clamp(i_2, min=0)
    i_2_neg = - torch.clamp(i_2, max=0)
    numerator = torch.minimum(i_1_pos, i_2_pos).sum() + torch.minimum(i_1_neg, i_2_neg).sum()
    denominator = torch.maximum(i_1_pos, i_2_pos).sum() + torch.maximum(i_1_neg, i_2_neg).sum()
    return numerator / (denominator + eps)


def get_iou_discrete(set_1, set_2):
    """
    Evaluate the IoU between two sets:
        IoU := |set_1 intersect set_2| / |set_1 union set_2|
    :param set_1:
    :param set_2:
    :return:
    """

    if len(set_1) + len(set_2) == 0:
        return 0

    intersection = []
    union = []
    for pattern in set_1:
        union.append(pattern)
        if pattern in set_2:
            intersection.append(pattern)
    for pattern in set_2:
        if pattern not in union:
            union.append(pattern)

    iou = len(intersection) / len(union)

    return iou



if __name__ == '__main__':
    # i_1 = torch.Tensor([1.0, -1.0, 1.0, -1.0])
    # i_2 = torch.Tensor([0.5, -0.5, -0.5, 0.5])
    i_1 = torch.randn(100)
    i_2 = torch.randn(100)
    print(get_iou(i_1, i_2))
    print(get_pos_neg_iou(i_1, i_2))
    print(get_overlap(i_1, i_2))