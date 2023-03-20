import math

import numpy as np
import torch
from tqdm import tqdm
from .interaction_utils import generate_all_masks, generate_subset_masks, generate_supset_masks


def get_reward2harsanyi_mat(all_masks):
    '''
    The transformation matrix (containing 0, 1, -1's) from reward to and-interaction (Harsanyi)
    :param all_masks: a bool matrix indicating all subset S's, with shape [2^n, n]
    :return: a matrix, with shape 2^n * 2^n
    '''
    # all_masks = torch.BoolTensor(generate_all_masks(dim))
    # all_masks = all_masks.cpu()
    device = all_masks.device
    n_masks, _ = all_masks.shape
    mat = []
    for i in range(n_masks):
        mask_S = all_masks[i]
        row = torch.zeros(n_masks, device=device)
        # ===============================================================================================
        # Note: I(S) = \sum_{L\subseteq S} (-1)^{s-l} v(L)
        mask_Ls, L_indices = generate_subset_masks(mask_S, all_masks)
        L_indices = (L_indices == True).nonzero(as_tuple=False)
        assert mask_Ls.shape[0] == L_indices.shape[0]
        row[L_indices] = torch.pow(-1., mask_S.sum() - mask_Ls.sum(dim=1)).unsqueeze(1)
        # ===============================================================================================
        mat.append(row.clone())
    mat = torch.stack(mat).float()
    return mat.to(device)


def get_harsanyi2reward_mat(all_masks):
    # all_masks = torch.BoolTensor(generate_all_masks(dim))
    # all_masks = all_masks.cpu()
    device = all_masks.device
    n_masks, _ = all_masks.shape
    mat = []
    for i in range(n_masks):
        mask_S = all_masks[i]
        row = torch.zeros(n_masks, device=device)
        # ================================================================================================
        # Note: v(S) = \sum_{L\subseteq S} I(S)
        mask_Ls, L_indices = generate_subset_masks(mask_S, all_masks)
        row[L_indices] = 1.
        # ================================================================================================
        mat.append(row.clone())
    mat = torch.stack(mat).float()
    return mat.to(device)


def get_harsanyi2shaptaylor_mat(all_masks, order):
    # all_masks = torch.BoolTensor(generate_all_masks(dim))
    # all_masks = all_masks.cpu()
    device = all_masks.device
    n_masks, _ = all_masks.shape
    mat = []
    for i in range(n_masks):
        mask_S = all_masks[i]
        order_S = int(mask_S.sum().item())
        row = torch.zeros(n_masks, device=device)
        # ===============================================================================================
        # Note:
        #  (i)   if |S| < order, STI(S) = I(S)
        #  (ii)  if |S| = order, STI(S) = \sum_{T\supseteq S} \binom{|T|}{|S|}^{-1} * I(T)
        #  (iii) if |S| > order, STI(S) = 0
        if order_S < order:
            row[i] = 1.
        elif order_S > order:
            pass
        else:  # |S| == order
            mask_Ts, T_indices = generate_supset_masks(mask_S, all_masks)
            coefs = [1 / math.comb(int(mask_T.sum().item()), order_S) for mask_T in mask_Ts]
            row[T_indices] = torch.tensor(coefs, device=device)
        # ===============================================================================================
        mat.append(row.clone())
    mat = torch.stack(mat).float()
    return mat.to(device)


def get_harsanyi2shapinteraction_mat(all_masks):
    # all_masks = torch.BoolTensor(generate_all_masks(dim))
    # all_masks = all_masks.cpu()
    device = all_masks.device
    n_masks, dim = all_masks.shape
    # assert order <= dim, f"invalid order: {order} (max {dim})"
    mat = []
    for i in range(n_masks):
        mask_S = all_masks[i]
        order_S = int(mask_S.sum().item())
        row = torch.zeros(n_masks).to(device)
        # ===============================================================================================
        # Note: SI(S) = \sum_{T\supseteq S} (|T|-|S|+1)^{-1} * I(T)
        mask_Ts, T_indices = generate_supset_masks(mask_S, all_masks)
        coefs = [1 / (int(mask_T.sum().item()) - order_S + 1) for mask_T in mask_Ts]
        row[T_indices] = torch.tensor(coefs)
        # ===============================================================================================
        mat.append(row.clone())
    mat = torch.stack(mat).float()
    return mat.to(device)



if __name__ == '__main__':
    all_masks = torch.BoolTensor(generate_all_masks(12))
    get_harsanyi2reward_mat(all_masks)

    dim = 3
    all_masks = torch.BoolTensor(generate_all_masks(dim))
    reward2harsanyi = get_reward2harsanyi_mat(all_masks)
    harsanyi2sti_2 = get_harsanyi2shaptaylor_mat(all_masks, 2)
    harsanyi2sti_3 = get_harsanyi2shaptaylor_mat(all_masks, 3)
    harsanyi2si = get_harsanyi2shapinteraction_mat(all_masks)
    rewards = torch.randn(all_masks.shape[0])
    harsanyi = torch.matmul(reward2harsanyi, rewards)
    sti_2 = torch.matmul(harsanyi2sti_2, harsanyi)
    sti_3 = torch.matmul(harsanyi2sti_3, harsanyi)
    si = torch.matmul(harsanyi2si, harsanyi)
    print(all_masks.sum(dim=1).shape, all_masks.sum(dim=1))
    print(rewards[-1], sti_2.sum(), sti_3.sum(), harsanyi.sum(), si.sum())
    print(si[all_masks.sum(dim=1) == 1].sum(), rewards[-1] - rewards[0])
