import torch
from tqdm import tqdm


def generate_subset_masks(set_mask, all_masks):
    '''
    For a given S, generate its subsets L's, as well as the indices of L's in [all_masks]
    :param set_mask:
    :param all_masks:
    :return: the subset masks, the bool indice
    '''
    set_mask_ = set_mask.expand_as(all_masks)
    is_subset = torch.logical_or(set_mask_, torch.logical_not(all_masks))
    is_subset = torch.all(is_subset, dim=1)
    return all_masks[is_subset], is_subset


def get_harsanyi2reward_mat(all_masks):
    # all_masks = torch.BoolTensor(generate_all_masks(dim))
    # all_masks = all_masks.cpu()
    device = all_masks.device
    n_masks, _ = all_masks.shape
    mat = []
    for i in range(n_masks):
        mask_S = all_masks[i]
        row = torch.zeros(n_masks, device=device)
        # ============================================================
        # Note: v(S) = \sum_{L\subseteq S} I(S)
        mask_Ls, L_indices = generate_subset_masks(mask_S, all_masks)
        row[L_indices] = 1.
        # ============================================================
        mat.append(row.clone())
    mat = torch.stack(mat).float()
    return mat.to(device)


def _eval_approx_error(rewards, harsanyi, harsanyi2reward):
    """
    计算拟合误差 err := \sum_{S\subseteq N} [v(S) - g(S)] 其中
               g(S)=\sum_{S'\subseteq S, S'\ retained} I(S')
    :param rewards: [2^n, ]
    :param harsanyi: [2^n, ]
    :param harsanyi2reward: harsanyi -> reward 转换矩阵 [2^n, 2^n]
    :return:
    """
    return torch.sum(torch.square(rewards - torch.matmul(harsanyi2reward, harsanyi)))


def _test_remove_one(rewards, harsanyi, harsanyi2reward, to_remove):
    harsanyi_ = harsanyi.clone()
    harsanyi_[to_remove] = 0.
    error = _eval_approx_error(rewards, harsanyi_, harsanyi2reward)
    return error


def eval_pattern_num_approx_error_relation(harsanyi, masks, n_greedy=40):
    """
    :param harsanyi: [2^n, ]
    :param masks: [2^n, n] bool matrix
    :param n_greedy: 每删除一个 causal pattern 时，从强度最小的 `n_greedy` 个 pattern 中选择
                     特别地，n_greedy=1 时退化为每次选择强度最小的 pattern 删除
    :return: Tuple[List[int]: 保留的 pattern 数量, List[float]: 拟合误差]
    """
    device = harsanyi.device
    harsanyi = harsanyi.clone()
    harsanyi = harsanyi.to(device)
    masks = masks.to(device)

    harsanyi2reward = get_harsanyi2reward_mat(all_masks=masks)
    harsanyi2reward = harsanyi2reward.to(device)

    rewards = torch.matmul(harsanyi2reward, harsanyi)
    n_all_patterns = harsanyi.shape[0]
    harsanyi_order = torch.argsort(torch.abs(harsanyi)).tolist()  # from low-strength to high-strength

    pattern_num_list = [n_all_patterns]
    approx_error_list = [_eval_approx_error(rewards, harsanyi, harsanyi2reward).item()]

    for n_remove in tqdm(range(n_all_patterns), ncols=100, desc="evaluating"):
        candidates = harsanyi_order[:n_greedy]

        errors = torch.tensor([_test_remove_one(rewards, harsanyi, harsanyi2reward, to_remove)
                               for to_remove in candidates])

        error, to_remove_idx = torch.min(errors, dim=0)
        error, to_remove_idx = error.item(), to_remove_idx.item()
        to_remove = candidates[to_remove_idx]

        harsanyi[to_remove] = 0.
        harsanyi_order.remove(to_remove)

        pattern_num_list.append(n_all_patterns - n_remove)
        approx_error_list.append(error)

    return pattern_num_list[::-1], approx_error_list[::-1]


if __name__ == '__main__':
    import numpy as np
    import os
    import os.path as osp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


    # folder = "/data2/limingjie/InteractionAOG/saved-CI-remove-noisy-0.1/mlp5_commercial-0.01_1_200_adv0.01-20-0.1_0" \
    #          "/baseline_custom_single_ci_precise_vfunc_gt-log-odds" \
    #          "/finetune_lr_0.001_max_iter_50_suppress_1.0_bound_0.1" \
    #          "/eval-20-samples/train/class-0/sample-0000"
    folder = "./"
    harsanyi = torch.from_numpy(np.load(osp.join(folder, "harsanyi.npy"))).float().cuda()
    masks = torch.from_numpy(np.load(osp.join(folder, "masks.npy"))).bool().cuda()

    X, Y = eval_pattern_num_approx_error_relation(harsanyi, masks, n_greedy=1)

    plt.figure()
    plt.plot(X[20:], Y[20:])
    plt.title("causal pattern number -- approximation error")
    plt.xlabel("# causal patterns retained")
    plt.ylabel(r"$\sum_{S\subseteq N} [v(S) - g(S)]$ where "
               r"$g(S)=\sum_{S'\subseteq S, S'\ retained} I(S')$")
    plt.tight_layout()
    plt.savefig("relation.png")

