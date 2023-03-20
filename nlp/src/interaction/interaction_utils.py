import torch
import torch.nn as nn
import numpy as np
from pprint import pprint


def generate_all_masks(length: int) -> list:
    masks = list(range(2**length))
    masks = [np.binary_repr(mask, width=length) for mask in masks]
    masks = [[bool(int(item)) for item in mask] for mask in masks]
    return masks


def set_to_index(A):
    '''
    convert a boolean mask to an index
    :param A: <np.ndarray> bool (n_dim,)
    :return: an index

    [In] set_to_index(np.array([1, 0, 0, 1, 0]).astype(bool))
    [Out] 18
    '''
    assert len(A.shape) == 1
    A_ = A.astype(int)
    return np.sum([A_[-i-1] * (2 ** i) for i in range(A_.shape[0])])


def is_A_subset_B(A, B):
    '''
    Judge whether $A \subseteq B$ holds
    :param A: <numpy.ndarray> bool (n_dim, )
    :param B: <numpy.ndarray> bool (n_dim, )
    :return: Bool
    '''
    assert A.shape[0] == B.shape[0]
    return np.all(np.logical_or(np.logical_not(A), B))


def is_A_subset_Bs(A, Bs):
    '''
    Judge whether $A \subseteq B$ holds for each $B$ in 'Bs'
    :param A: <numpy.ndarray> bool (n_dim, )
    :param Bs: <numpy.ndarray> bool (n, n_dim)
    :return: Bool
    '''
    assert A.shape[0] == Bs.shape[1]
    is_subset = np.all(np.logical_or(np.logical_not(A), Bs), axis=1)
    return is_subset


def select_subset(As, B):
    '''
    Select A from As that satisfies $A \subseteq B$
    :param As: <numpy.ndarray> bool (n, n_dim)
    :param B: <numpy.ndarray> bool (n_dim, )
    :return: a subset of As
    '''
    assert As.shape[1] == B.shape[0]
    is_subset = np.all(np.logical_or(np.logical_not(As), B), axis=1)
    return As[is_subset]

def get_subset(A):
    '''
    Generate the subset of A
    :param A: <numpy.ndarray> bool (n_dim, )
    :return: subsets of A
    '''
    assert len(A.shape) == 1
    n_dim = A.shape[0]
    n_subsets = 2 ** A.sum()
    subsets = np.zeros(shape=(n_subsets, n_dim)).astype(bool)
    subsets[:, A] = np.array(generate_all_masks(A.sum()))
    return subsets


def generate_subset_masks(set_mask, all_masks):
    '''

    :param set_mask:
    :param all_masks:
    :return: the subset masks, the bool indice
    '''
    set_mask_ = set_mask.expand_as(all_masks)
    is_subset = torch.logical_or(set_mask_, torch.logical_not(all_masks))
    is_subset = torch.all(is_subset, dim=1)
    return all_masks[is_subset], is_subset



def calculate_all_subset_outputs_pytorch(
    model: nn.Module,
    input: torch.Tensor,
    baseline: torch.Tensor
) -> (torch.Tensor, torch.Tensor):
    '''
    This function returns the output of all possible subsets of the input
    :param model: the target model
    :param input: [1, sentence_len, emb_dim] (batch_first)
    :param baseline: [1, sentence_len, emb_dim] the same as input
    :return: masks and the outputs
    '''
    assert torch.all(torch.eq(torch.tensor(input.shape), torch.tensor(baseline.shape)))
    bs, sentence_len, emb_dim = input.shape
    assert bs == 1
    device = input.device
    ############################## This part different ##############################################
    masks = torch.BoolTensor(generate_all_masks(sentence_len)).to(device)  # [2^len, len]
    masks_ = torch.stack([masks] * emb_dim, dim=2)  # [2^len, len, emb_dim]
    masked_inputs = torch.where(masks_, input.expand_as(masks_), baseline.expand_as(masks_))
    with torch.no_grad():
        outputs = model.emb2out(masked_inputs)  # [2^len, out_dim]
    ############################## This part different ##############################################
    return masks, outputs



def calculate_all_subset_outputs(model, input, baseline):
    if isinstance(model, nn.Module):
        return calculate_all_subset_outputs_pytorch(model, input, baseline)
    else:
        raise NotImplementedError


def calculate_all_subset_coef_matrix(length: int) -> torch.Tensor:
    '''

    :param length: the number of input variables (feature dim)
    :return: np.ndarray (2^l, 2^l), each row -- the coefs (0/1/-1) of I(S) before each v(S')
    '''
    coefs = torch.zeros(2 ** length, 2 ** length)
    all_masks = torch.BoolTensor(generate_all_masks(length))

    for i in range(all_masks.shape[0]):
        mask = all_masks[i]
        subset_mask, subset_indice = generate_subset_masks(set_mask=mask, all_masks=all_masks)
        coef_S = torch.sum(subset_mask, dim=1) + torch.sum(mask)
        coefs[i][subset_indice] = torch.pow(-1, coef_S).float()

    return coefs





if __name__ == '__main__':
    emb_dim = 5
    sentence_len = 3
    # input = torch.randn(1, sentence_len, emb_dim)
    # baseline = torch.tensor([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]]).float()
    # masks, outputs = calculate_all_subset_outputs_pytorch(None, input, baseline)
    # print(outputs.shape)

    # all_masks = generate_all_masks(6)
    # all_masks = torch.BoolTensor(all_masks)
    # set_mask = torch.BoolTensor([1, 0, 1, 1, 0, 0])
    # print(generate_subset_masks(set_mask, all_masks))

    # print(calculate_all_subset_coef_matrix(4))
    # print(get_subset(np.array([1, 0, 0, 1, 0, 1]).astype(bool)))
    #
    # Bs = get_subset(np.array([1, 0, 0, 1, 0, 1]).astype(bool))
    # A = np.array([1, 0, 0, 1, 0, 0]).astype(bool)
    # print(is_A_subset_Bs(A, Bs))

    print(calculate_all_subset_coef_matrix(3))