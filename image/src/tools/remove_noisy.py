import numpy as np
import torch
from tqdm import tqdm
import sys
sys.path.append("..")

from harsanyi.and_or_harsanyi_utils import get_Iand2reward_mat


def eval_explain_ratio_v2_given_coalitions(CI, masks, selected):
    if torch.sum(selected) == 0: return 0.0

    if isinstance(CI, np.ndarray): CI = torch.FloatTensor(CI.copy())
    if isinstance(masks, np.ndarray): masks = torch.BoolTensor(masks.copy())
    if isinstance(selected, np.ndarray): selected = torch.BoolTensor(selected.copy())

    not_empty = torch.any(masks, dim=1)
    unselected = torch.logical_not(selected)

    numerator = torch.abs(CI[selected][not_empty[selected]]).sum()
    denominator = torch.abs(CI[selected][not_empty[selected]]).sum() + \
                  torch.abs(CI[unselected][not_empty[unselected]].sum()) + 1e-7
    ratio = numerator / denominator

    return ratio.item()


def get_retain_order(CI, masks, n_greedy=40, device=torch.device("cpu")):
    CI = CI.clone()
    CI = CI.to(device)
    masks = masks.to(device)

    harsanyi2reward = get_Iand2reward_mat(dim=masks.shape[1])
    harsanyi2reward = harsanyi2reward.to(device)

    rewards = torch.matmul(harsanyi2reward, CI)
    original_CI = CI.clone()
    n_all_patterns = CI.shape[0]
    CI_order = torch.argsort(torch.abs(CI)).tolist()  # from low-strength to high-strength
    removed_coalition_ids = []

    for n_remove in tqdm(range(n_all_patterns), ncols=100, desc="removing"):
        candidates = CI_order[:n_greedy]
        to_remove = candidates[0]
        CI_ = CI.clone()
        CI_[to_remove] = 0.
        error = torch.sum(torch.square(rewards - torch.matmul(harsanyi2reward, CI_)))

        for i in range(1, len(candidates)):
            candidate = candidates[i]
            CI_ = CI.clone()
            CI_[candidate] = 0.
            error_ = torch.sum(torch.square(rewards - torch.matmul(harsanyi2reward, CI_)))
            if error_ < error:
                to_remove = candidate
                error = error_

        CI[to_remove] = 0.
        CI_order.remove(to_remove)
        removed_coalition_ids.append(to_remove)

    return removed_coalition_ids[::-1]


def remove_noisy_greedy(rewards, CI, masks, harsanyi2reward, min_patterns, max_patterns=80, n_greedy=20,
                        thres_approx_error=0.1, thres_explain_ratio=0.95, thres_explain_ratio_v1=0.7):
    original_CI = CI.clone()
    CI = CI.clone()
    v_N = rewards[torch.all(masks, dim=1)].item()
    v_empty = rewards[torch.logical_not(torch.any(masks, dim=1))].item()
    device = rewards.device
    CI_order = torch.argsort(torch.abs(CI)).tolist()  # from low-strength to high-strength

    n_all_patterns = CI.shape[0]
    unremoved = torch.ones(n_all_patterns).bool().to(device)

    removed_coalition_ids = []
    errors = []
    explain_ratios = []
    explain_ratios_v1 = []

    for n_remove in tqdm(range(n_all_patterns - min_patterns), ncols=100, desc="removing"):
        candidates = CI_order[:n_greedy]
        to_remove = candidates[0]
        CI_ = CI.clone()
        CI_[to_remove] = 0.
        error = torch.sum(torch.square(rewards - torch.matmul(harsanyi2reward, CI_)))

        for i in range(1, len(candidates)):
            candidate = candidates[i]
            CI_ = CI.clone()
            CI_[candidate] = 0.
            error_ = torch.sum(torch.square(rewards - torch.matmul(harsanyi2reward, CI_)))
            if error_ < error:
                to_remove = candidate
                error = error_

        CI[to_remove] = 0.
        CI_order.remove(to_remove)
        unremoved[to_remove] = False
        removed_coalition_ids.append(to_remove)
        errors.append(error.item())
        explain_ratio = eval_explain_ratio_v2_given_coalitions(original_CI, masks, unremoved)
        explain_ratios.append(explain_ratio)
        explain_ratio_v1 = torch.abs(original_CI[unremoved]).sum() / torch.abs(original_CI).sum()
        explain_ratios_v1.append(explain_ratio_v1)

    assert len(errors) == len(removed_coalition_ids) and len(errors) == len(explain_ratios)
    errors = np.array(errors)
    explain_ratios = np.array(explain_ratios)
    explain_ratios_v1 = np.array(explain_ratios_v1)  # (deprecated)  ###############################
    print(torch.norm(rewards, p=2).item())
    normalized_errors = np.sqrt(errors) / torch.norm(rewards, p=2).item()

    satisfy = np.logical_and(explain_ratios > thres_explain_ratio, normalized_errors < thres_approx_error)
    # satisfy = np.logical_and(explain_ratios > thres_explain_ratio, explain_ratios_v1 > thres_explain_ratio_v1)
    last_satisfy = 0
    for i in range(len(satisfy) - 1, -1, -1):
        if satisfy[i]:
            last_satisfy = i
            break

    if n_all_patterns - last_satisfy - 1 > max_patterns:
        last_satisfy = n_all_patterns - max_patterns - 1

    final_removed_ids = removed_coalition_ids[:last_satisfy+1]
    final_retained_ids = [idx for idx in range(CI.shape[0]) if idx not in final_removed_ids]
    final_error = errors[last_satisfy]
    final_normalized_error = normalized_errors[last_satisfy]
    final_ratio = explain_ratios[last_satisfy]
    print(f"[n_greedy={n_greedy}] -- # coalitions: {len(final_retained_ids)} | error: {final_error:.4f} "
          f"| normalized error: {final_normalized_error:.4f} | ratio: {final_ratio:.4f}\n")

    verbose = {
        "errors": errors,
        "explain_ratios": explain_ratios,
        "final_retained": final_retained_ids,
        "final_error": final_error,
        "final_ratio": final_ratio
    }
    return unremoved, verbose



def remove_noisy_low_strength_first(rewards, CI, masks, harsanyi2reward, min_patterns):
    original_CI = CI.clone()
    device = rewards.device
    CI_order = torch.argsort(torch.abs(CI)).tolist()  # from low-strength to high-strength

    n_all_patterns = CI.shape[0]
    unremoved = torch.ones(n_all_patterns).bool().to(device)

    errors = []
    explain_ratios = []

    for n_remove in tqdm(range(n_all_patterns - min_patterns), ncols=100, desc="removing"):
        to_remove = CI_order[0]
        CI[to_remove] = 0.
        error = torch.sum(torch.square(rewards - torch.matmul(harsanyi2reward, CI)))

        CI_order.remove(to_remove)
        unremoved[to_remove] = False
        errors.append(error)
        explain_ratio = eval_explain_ratio_v2_given_coalitions(original_CI, masks, unremoved)
        explain_ratios.append(explain_ratio)

    verbose = {
        "errors": errors,
        "explain_ratios": explain_ratios
    }
    return unremoved, verbose

