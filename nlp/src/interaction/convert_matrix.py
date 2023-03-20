import torch
from tqdm import tqdm
from .interaction_utils import generate_all_masks, generate_subset_masks


HIGH_DIM_THRES = 13


def get_reward2harsanyi_mat(dim):
    if dim > HIGH_DIM_THRES:
        return get_reward2harsanyi_mat_sparse(dim)
    else:
        return get_reward2harsanyi_mat_dense(dim)


def get_reward2harsanyi_mat_dense(dim):
    '''
    The transformation matrix (containing 0, 1, -1's) from reward to and-interaction (Harsanyi)
    :param dim: the input dimension n
    :return: a matrix, with shape 2^n * 2^n
    '''
    all_masks = torch.BoolTensor(generate_all_masks(dim))
    n_masks, _ = all_masks.shape
    mat = []
    for i in range(n_masks):
        mask_S = all_masks[i]
        row = torch.zeros(n_masks)
        # ===============================================================================================
        # Note: I(S) = \sum_{L\subseteq S} (-1)^{s-l} v(L)
        mask_Ls, L_indices = generate_subset_masks(mask_S, all_masks)
        L_indices = (L_indices == True).nonzero(as_tuple=False)
        assert mask_Ls.shape[0] == L_indices.shape[0]
        row[L_indices] = torch.pow(-1., mask_S.sum() - mask_Ls.sum(dim=1)).unsqueeze(1)
        # ===============================================================================================
        mat.append(row.clone())
    mat = torch.stack(mat).float()
    return mat

# ===========================
# add sparse matrix support
# ===========================
def get_reward2harsanyi_mat_sparse(dim):
    '''
    The transformation matrix (containing 0, 1, -1's) from reward to and-interaction (Harsanyi)
    :param dim: the input dimension n
    :return: a matrix, with shape 2^n * 2^n
    '''
    all_masks = torch.BoolTensor(generate_all_masks(dim))
    n_masks, _ = all_masks.shape
    mat = {}
    for i in tqdm(range(n_masks), ncols=100, desc="[v->I^and] Generating mask"):
        mask_S = all_masks[i]
        row = torch.zeros(n_masks)
        # ===============================================================================================
        # Note: I(S) = \sum_{L\subseteq S} (-1)^{s-l} v(L)
        mask_Ls, L_indices = generate_subset_masks(mask_S, all_masks)
        L_indices = (L_indices == True).nonzero(as_tuple=False)
        assert mask_Ls.shape[0] == L_indices.shape[0]
        row[L_indices] = torch.pow(-1., mask_S.sum() - mask_Ls.sum(dim=1)).unsqueeze(1)
        # ===============================================================================================
        for j in L_indices[:, 0]:
            mat[(i, j)] = row[j]

    mat = torch.sparse_coo_tensor(
        indices=torch.tensor(list(mat.keys())).t(),
        values=torch.tensor(list(mat.values())),
        size=(n_masks, n_masks)
    ).float()
    return mat


def get_harsanyi2reward_mat(dim):
    all_masks = torch.BoolTensor(generate_all_masks(dim))
    n_masks, _ = all_masks.shape
    mat = []
    for i in range(n_masks):
        mask_S = all_masks[i]
        row = torch.zeros(n_masks)
        # ================================================================================================
        # Note: v(S) = \sum_{L\subseteq S} I(S)
        mask_Ls, L_indices = generate_subset_masks(mask_S, all_masks)
        row[L_indices] = 1.
        # ================================================================================================
        mat.append(row.clone())
    mat = torch.stack(mat).float()
    return mat


if __name__ == '__main__':
    print(get_reward2harsanyi_mat(3))