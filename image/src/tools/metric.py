import torch
import numpy as np
import os
import os.path as osp
import sys


def eval_explain_ratio_v1(CI, masks, n_context):
    if n_context == 0: return 0.0

    if isinstance(CI, np.ndarray): CI = torch.FloatTensor(CI.copy())
    if isinstance(masks, np.ndarray): masks = torch.BoolTensor(masks.copy())

    not_empty = torch.any(masks, dim=1)
    CI_order = torch.argsort(-torch.abs(CI))  # strength of interaction: from high -> low
    # ratio = torch.abs(CI[CI_order][:n_context][not_empty[CI_order][:n_context]]).sum() / (torch.abs(CI[not_empty]).sum() + 1e-7)
    # ratio = torch.abs(CI[CI_order][n_context:][not_empty[CI_order][n_context:]].sum()) / (torch.abs(CI[not_empty].sum()) + 1e-7)
    # ratio = torch.abs(CI[CI_order][n_context:].sum()) / (torch.abs(CI.sum()) + 1e-7)
    # ratio = 1 - ratio

    # the original one
    # ratio = torch.abs(CI[CI_order][:n_context][not_empty[CI_order][:n_context]]).sum() / (torch.abs(CI[not_empty]).sum() + 1e-7)

    # # the revised version (09-15)
    # numerator = torch.abs(CI[CI_order][:n_context][not_empty[CI_order][:n_context]]).sum()
    # denominator = torch.abs(CI[CI_order][:n_context][not_empty[CI_order][:n_context]]).sum() +\
    #               torch.abs(CI[CI_order][n_context:][not_empty[CI_order][n_context:]].sum()) + 1e-7
    # ratio = numerator / denominator

    # the revised version (09-22)
    v_empty = CI[torch.logical_not(not_empty)].item()
    denominator = torch.abs(CI).sum() + 1e-7
    numerator = torch.abs(CI[CI_order][:n_context][not_empty[CI_order][:n_context]]).sum() + abs(v_empty)
    ratio = numerator / denominator

    return ratio.item()


def eval_explain_ratio_v2(CI, masks, n_context):
    if n_context == 0: return 0.0

    if isinstance(CI, np.ndarray): CI = torch.FloatTensor(CI.copy())
    if isinstance(masks, np.ndarray): masks = torch.BoolTensor(masks.copy())

    not_empty = torch.any(masks, dim=1)
    CI_order = torch.argsort(-torch.abs(CI))  # strength of interaction: from high -> low
    # ratio = torch.abs(CI[CI_order][:n_context][not_empty[CI_order][:n_context]]).sum() / (torch.abs(CI[not_empty]).sum() + 1e-7)
    # ratio = torch.abs(CI[CI_order][n_context:][not_empty[CI_order][n_context:]].sum()) / (torch.abs(CI[not_empty].sum()) + 1e-7)
    # ratio = torch.abs(CI[CI_order][n_context:].sum()) / (torch.abs(CI.sum()) + 1e-7)
    # ratio = 1 - ratio

    # the original one
    # ratio = torch.abs(CI[CI_order][:n_context][not_empty[CI_order][:n_context]]).sum() / (torch.abs(CI[not_empty]).sum() + 1e-7)

    # the revised version (09-15)
    numerator = torch.abs(CI[CI_order][:n_context][not_empty[CI_order][:n_context]]).sum()
    denominator = torch.abs(CI[CI_order][:n_context][not_empty[CI_order][:n_context]]).sum() + \
                  torch.abs(CI[CI_order][n_context:][not_empty[CI_order][n_context:]].sum()) + 1e-7
    ratio = numerator / denominator

    return ratio.item()


def eval_iou_discrete(gt_patterns, interaction, masks):
    actual_patterns = []
    for ii in range(len(gt_patterns)):
        interaction_order = np.argsort(-np.abs(interaction))
        actual_pattern = np.arange(masks.shape[1])[masks[interaction_order][ii]]
        actual_patterns.append(actual_pattern.tolist())

    intersection = []
    union = []
    for pattern in actual_patterns:
        union.append(pattern)
        if pattern in gt_patterns:
            intersection.append(pattern)
    for pattern in gt_patterns:
        if pattern not in union:
            union.append(pattern)

    iou = len(intersection) / len(union)

    return iou


def eval_explain_ratio_v2_given_coalition_ids(CI, masks, selected_ids):
    if len(selected_ids) == 0: return 0.0

    selected = torch.zeros(CI.shape[0]).bool()
    selected[selected_ids] = True

    if isinstance(CI, np.ndarray): CI = torch.FloatTensor(CI.copy())
    if isinstance(masks, np.ndarray): masks = torch.BoolTensor(masks.copy())

    not_empty = torch.any(masks, dim=1)
    unselected = torch.logical_not(selected)

    numerator = torch.abs(CI[selected][not_empty[selected]]).sum()
    denominator = torch.abs(CI[selected][not_empty[selected]]).sum() + \
                  torch.abs(CI[unselected][not_empty[unselected]].sum()) + 1e-7
    ratio = numerator / denominator

    return ratio.item()


def eval_explain_ratio_v1_given_coalition_ids(CI, masks, selected_ids):
    if len(selected_ids) == 0: return 0.0

    selected = torch.zeros(CI.shape[0]).bool()
    selected[selected_ids] = True

    if isinstance(CI, np.ndarray): CI = torch.FloatTensor(CI.copy())
    if isinstance(masks, np.ndarray): masks = torch.BoolTensor(masks.copy())

    not_empty = torch.any(masks, dim=1)
    unselected = torch.logical_not(selected)
    v_empty = CI[torch.logical_not(not_empty)].item()

    numerator = torch.abs(CI[selected][not_empty[selected]]).sum() + abs(v_empty)
    denominator = torch.abs(CI).sum().item()
    ratio = numerator / denominator

    return ratio.item()