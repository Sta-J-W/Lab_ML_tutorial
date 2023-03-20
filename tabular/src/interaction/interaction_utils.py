import torch
import torch.nn as nn
import numpy as np
from pprint import pprint
import doctest


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


def set_minus(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    '''
    calculate A/B
    :param A: <numpy.ndarray> bool (n_dim, )
    :param B: <numpy.ndarray> bool (n_dim, )
    :return: A\B

    >>> set_minus(A=np.array([1, 1, 0, 0, 1, 1], dtype=bool), B=np.array([1, 0, 0, 0, 0, 1], dtype=bool))
    array([False,  True, False, False,  True, False])

    >>> set_minus(A=np.array([1, 1, 0, 0, 1, 1], dtype=bool), B=np.array([1, 0, 1, 0, 0, 1], dtype=bool))
    array([False,  True, False, False,  True, False])
    '''
    assert A.shape[0] == B.shape[0] and len(A.shape) == 1 and len(B.shape) == 1
    A_ = A.copy()
    A_[B] = False
    return A_


def get_subset(A):
    '''
    Generate the subset of A
    :param A: <numpy.ndarray> bool (n_dim, )
    :return: subsets of A

    >>> get_subset(np.array([1, 0, 0, 1, 0, 1], dtype=bool))
    array([[False, False, False, False, False, False],
           [False, False, False, False, False,  True],
           [False, False, False,  True, False, False],
           [False, False, False,  True, False,  True],
           [ True, False, False, False, False, False],
           [ True, False, False, False, False,  True],
           [ True, False, False,  True, False, False],
           [ True, False, False,  True, False,  True]])
    '''
    assert len(A.shape) == 1
    n_dim = A.shape[0]
    n_subsets = 2 ** A.sum()
    subsets = np.zeros(shape=(n_subsets, n_dim)).astype(bool)
    subsets[:, A] = np.array(generate_all_masks(A.sum()))
    return subsets


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


def generate_supset_masks(set_mask, all_masks):
    '''
    For a given S, generate its supsets T's, as well as the indices of T's in [all_masks]
    :param set_mask:
    :param all_masks:
    :return: the subset masks, the bool indice

    >>> all_masks = torch.BoolTensor(generate_all_masks(5))
    >>> set_mask = torch.BoolTensor([0, 1, 0, 1, 0])
    >>> generate_supset_masks(set_mask, all_masks)
    (tensor([[False,  True, False,  True, False],
            [False,  True, False,  True,  True],
            [False,  True,  True,  True, False],
            [False,  True,  True,  True,  True],
            [ True,  True, False,  True, False],
            [ True,  True, False,  True,  True],
            [ True,  True,  True,  True, False],
            [ True,  True,  True,  True,  True]]), tensor([False, False, False, False, False, False, False, False, False, False,
             True,  True, False, False,  True,  True, False, False, False, False,
            False, False, False, False, False, False,  True,  True, False, False,
             True,  True]))
    '''
    set_mask_ = set_mask.expand_as(all_masks)
    is_supset = torch.logical_or(all_masks, torch.logical_not(set_mask_))
    is_supset = torch.all(is_supset, dim=1)
    return all_masks[is_supset], is_supset


def generate_reverse_subset_masks(set_mask, all_masks):
    '''
    For a given S, with subsets L's, generate N\L, as well as the indices of L's in [all_masks]
    :param set_mask:
    :param all_masks:
    :return:
    '''
    set_mask_ = set_mask.expand_as(all_masks)
    is_rev_subset = torch.logical_or(set_mask_, all_masks)
    is_rev_subset = torch.all(is_rev_subset, dim=1)
    return all_masks[is_rev_subset], is_rev_subset


def generate_set_with_intersection_masks(set_mask, all_masks):
    '''
    For a given S, generate L's, s.t. L and S have intersection as well as the indices of L's in [all_masks]
    :param set_mask:
    :param all_masks:
    :return:
    '''
    set_mask_ = set_mask.expand_as(all_masks)
    have_intersection = torch.logical_and(set_mask_, all_masks)
    have_intersection = torch.any(have_intersection, dim=1)
    return all_masks[have_intersection], have_intersection



def calculate_all_subset_outputs_pytorch(
    model: nn.Module,
    input: torch.Tensor,
    baseline: torch.Tensor
) -> (torch.Tensor, torch.Tensor):
    '''
    This function returns the output of all possible subsets of the input
    :param model: the target model
    :param input: a single input vector (for tabular data) ...
    :param baseline: the baseline in each dimension
    :return: masks and the outputs
    '''
    assert len(input.shape) == 1
    n_attributes = input.shape[0]
    device = input.device
    masks = torch.BoolTensor(generate_all_masks(n_attributes)).to(device)
    masked_inputs = torch.where(masks, input.expand_as(masks), baseline.expand_as(masks))
    # print(masked_inputs)
    with torch.no_grad():
        outputs = model(masked_inputs)
    return masks, outputs


def calculate_all_subset_outputs_function(
    model,
    input: torch.Tensor,
    baseline: torch.Tensor
) -> (torch.Tensor, torch.Tensor):
    '''
    This function returns the output of all possible subsets of the input
    :param model: the target model
    :param input: a single input vector (for tabular data) ...
    :param baseline: the baseline in each dimension
    :return: masks and the outputs
    '''
    assert len(input.shape) == 1
    n_attributes = input.shape[0]
    device = input.device
    masks = torch.BoolTensor(generate_all_masks(n_attributes)).to(device)
    masked_inputs = torch.where(masks, input.expand_as(masks), baseline.expand_as(masks))
    with torch.no_grad():
        outputs = model(masked_inputs)
    return masks, outputs


def calculate_all_subset_outputs(model, input, baseline):
    if isinstance(model, nn.Module):
        return calculate_all_subset_outputs_pytorch(model, input, baseline)
    elif str(type(model)) == "<class 'function'>":
        return calculate_all_subset_outputs_function(model, input, baseline)
    elif "AndSum" in str(type(model)):
        return calculate_all_subset_outputs_function(model, input, baseline)
    else:
        raise NotImplementedError(f"Unexpected model type: {type(model)}")




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
    # dim = 5
    # input = torch.randn(dim)
    # baseline = torch.FloatTensor([float(100 + 100 * i) for i in range(dim)])
    # model = nn.Linear(dim, 2)
    # calculate_all_subset_outputs_pytorch(model, input, baseline)

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

    # all_masks = generate_all_masks(12)
    # all_masks = np.array(all_masks, dtype=bool)
    # set_index_list = []
    # for mask in all_masks:
    #     set_index_list.append(set_to_index(mask))
    # print(len(set_index_list), len(set(set_index_list)))
    # print(min(set_index_list), max(set_index_list))

    import doctest
    doctest.testmod()



    # S [1 0 0 1 0] subset(S) -> [4, 5]