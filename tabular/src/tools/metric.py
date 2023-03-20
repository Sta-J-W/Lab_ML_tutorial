import torch
import numpy as np
import os
import os.path as osp
import sys
sys.path.append("..")

from interaction.conditioned_interaction import get_reward
from interaction.interaction_utils import generate_subset_masks


def eval_overall_objectiveness_ours(model, X, y, baseline, selected_dim, CI, masks, **kwargs):
    model.eval()
    with torch.no_grad():
        v_N = get_reward(values=model(X.unsqueeze(0)), selected_dim=selected_dim, **kwargs).item()
        err = []
        for S in masks:
            # calculate v(S) here
            x_S = torch.where(S, X, baseline)
            v_S = get_reward(values=model(x_S.unsqueeze(0)), selected_dim=selected_dim, **kwargs).item()
            # calculate g(S) here
            _, indice = generate_subset_masks(S, masks)
            g_S = CI[indice].sum().item()
            # store the error
            err.append(np.abs(v_S - g_S))
    err = np.array(err)
    overall_nonobj = np.mean(err) / np.abs(v_N)
    return overall_nonobj


def eval_overall_objectiveness_shapley_value(model, X, y, baseline, selected_dim, attribution, all_masks, **kwargs):
    model.eval()
    with torch.no_grad():
        v_N = get_reward(values=model(X.unsqueeze(0)), selected_dim=selected_dim, **kwargs).item()
        v_empty = get_reward(values=model(baseline.unsqueeze(0)), selected_dim=selected_dim, **kwargs).item()
        err = []
        for S in all_masks:
            # calculate v(S) here
            x_S = torch.where(S, X, baseline)
            v_S = get_reward(values=model(x_S.unsqueeze(0)), selected_dim=selected_dim, **kwargs).item()
            # calculate g(S) here
            g_S = v_empty + attribution[S].sum()
            g_S = g_S.item()
            # store the error
            err.append(np.abs(v_S - g_S))
    err = np.array(err)
    overall_nonobj = np.mean(err) / np.abs(v_N)
    return overall_nonobj


def eval_overall_objectiveness_attribution_method(model, X, y, baseline, selected_dim, attribution, all_masks, **kwargs):
    model.eval()
    with torch.no_grad():
        v_N = get_reward(values=model(X.unsqueeze(0)), selected_dim=selected_dim, **kwargs).item()
        err = []
        for S in all_masks:
            # calculate v(S) here
            x_S = torch.where(S, X, baseline)
            v_S = get_reward(values=model(x_S.unsqueeze(0)), selected_dim=selected_dim, **kwargs).item()
            # calculate g(S) here
            g_S = attribution[S].sum()
            g_S = g_S.item()
            # store the error
            err.append(np.abs(v_S - g_S))
    err = np.array(err)
    overall_nonobj = np.mean(err) / np.abs(v_N)
    return overall_nonobj


def eval_overall_objectiveness_student_model(teacher, X, y, baseline, selected_dim, student, all_masks, **kwargs):
    teacher.eval()
    with torch.no_grad():
        v_N = get_reward(values=teacher(X.unsqueeze(0)), selected_dim=selected_dim, **kwargs).item()
        err = []
        for S in all_masks:
            # the masked input
            x_S = torch.where(S, X, baseline)
            # calculate v(S) here
            v_S = get_reward(values=teacher(x_S.unsqueeze(0)), selected_dim=selected_dim, **kwargs).item()
            # calculate g(S) here
            g_S = get_reward(values=student(x_S.unsqueeze(0)), selected_dim=selected_dim, **kwargs).item()
            # store the error
            err.append(np.abs(v_S - g_S))
    err = np.array(err)
    overall_nonobj = np.mean(err) / np.abs(v_N)
    return overall_nonobj


def calculate_all_v_S(model, X, y, baseline, selected_dim, masks, **kwargs):
    # model.eval()
    v_S_list = []
    with torch.no_grad():
        for S in masks:
            # calculate v(S) here
            x_S = torch.where(S, X, baseline)
            v_S = get_reward(values=model(x_S.unsqueeze(0)), selected_dim=selected_dim, **kwargs)
            v_S_list.append(v_S)
    return torch.cat(v_S_list)


def calculate_CI_to_g_S_mat(masks):
    mat = []
    for idx in range(masks.shape[0]):
        S = masks[idx]
        _, indice = generate_subset_masks(S, masks)
        mat.append(indice.clone())
    mat = torch.stack(mat).float()
    return mat


def eval_objectiveness_ours_given_v_S(v_S_list, v_N, CI, masks, **kwargs):
    err = []
    for idx in range(masks.shape[0]):
        S = masks[idx]
        # calculate v(S) here
        v_S = v_S_list[idx]
        # calculate g(S) here
        _, indice = generate_subset_masks(S, masks)
        g_S = CI[indice].sum().item()
        # store the error
        err.append(np.abs(v_S - g_S))
    err = np.array(err)
    overall_nonobj = np.mean(err) / np.abs(v_N)
    return overall_nonobj


def eval_objectiveness_given_v_S_g_S(v_S_list, g_S_list, v_N):
    mean_err = torch.abs(v_S_list - g_S_list).mean().item()
    overall_nonobj = mean_err / np.abs(v_N)
    return overall_nonobj

def eval_ratio_objectiveness_relation_ours(model, X, y, baseline, selected_dim, CI, masks, output_file, **kwargs):
    pass

def eval_ratio_pattern_relation_ours(CI, masks, output_file, **kwargs):
    pass


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