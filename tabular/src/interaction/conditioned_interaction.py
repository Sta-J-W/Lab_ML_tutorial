'''
We define the conditioned interaction (only consider ineractions within S):

$$ CI(S)=I^S(S) = \sum_{L\subset S} (-1)^{(s-l)} f(x_L) $$

The conditioned interaction satisfies the efficiency property:

$$ f(x_N) = f(x_\emptyset) + \sum_{S\subset N,S\neq\emptyset}CI(S) $$

'''

from tqdm import tqdm
from .interaction_utils import *


def get_reward(values, selected_dim, **kwargs):
    if selected_dim == "max":
        values = values[:, torch.argmax(values[-1])]  # select the predicted dimension, by default
    elif selected_dim == "0":
        values = values[:, 0]
    elif selected_dim == "gt":
        assert "gt" in kwargs.keys()
        gt = kwargs["gt"]
        values = values[:, gt]  # select the ground-truth dimension
    elif selected_dim == "gt-log-odds":
        assert "gt" in kwargs.keys()
        gt = kwargs["gt"]
        eps = 1e-7
        values = torch.softmax(values, dim=1)
        values = values[:, gt]
        values = torch.log(values / (1 - values + eps) + eps)
    elif selected_dim == "max-log-odds":
        eps = 1e-7
        values = torch.softmax(values, dim=1)
        values = values[:, torch.argmax(values[-1])]
        values = torch.log(values / (1 - values + eps) + eps)
    elif selected_dim == "gt-logistic-odds":
        assert "gt" in kwargs.keys()
        gt = kwargs["gt"]
        assert gt == 0 or gt == 1
        eps = 1e-7
        assert len(values.shape) == 2 and values.shape[1] == 1
        values = torch.sigmoid(values)[:, 0]
        if gt == 0:
            values = 1 - values
        else:
            values = values
        values = torch.log(values / (1 - values + eps) + eps)
    elif selected_dim == "logistic-odds":
        eps = 1e-7
        assert len(values.shape) == 2 and values.shape[1] == 1
        values = torch.sigmoid(values)[:, 0]
        if torch.round(values[-1]) == 0.:
            values = 1 - values
        else:
            values = values
        values = torch.log(values / (1 - values + eps) + eps)
    elif selected_dim == "gt-prob-log-odds":
        assert "gt" in kwargs.keys()
        gt = kwargs["gt"]
        assert gt == 0 or gt == 1
        eps = 1e-7
        assert len(values.shape) == 2 and values.shape[1] == 1
        values = values[:, 0]
        if gt == 0:
            values = 1 - values
        else:
            values = values
        values = torch.log(values / (1 - values + eps) + eps)
    elif selected_dim == "prob-log-odds":
        eps = 1e-7
        assert len(values.shape) == 2 and values.shape[1] == 1
        values = values[:, 0]
        if torch.round(values[-1]) == 0.:
            values = 1 - values
        else:
            values = values
        values = torch.log(values / (1 - values + eps) + eps)
    elif selected_dim == None:
        values = values
    else:
        raise Exception(f"Unknown [selected_dim] {selected_dim}.")

    return values


def calculate_precise_CI(masks, values, selected_dim="max", **kwargs):
    '''
    :param masks:
    :param values:
    :param selected_dim:
    :return:
    '''
    device = values.device
    values = get_reward(values, selected_dim, **kwargs)
    # if selected_dim == "max":
    #     values = values[:, torch.argmax(values[-1])]  # select the predicted dimension, by default
    # elif selected_dim == "0":
    #     values = values[:, 0]
    # elif selected_dim == "gt":
    #     assert "gt" in kwargs.keys()
    #     gt = kwargs["gt"]
    #     values = values[:, gt]  # select the ground-truth dimension
    # elif selected_dim == "gt-log-odds":
    #     assert "gt" in kwargs.keys()
    #     gt = kwargs["gt"]
    #     eps = 1e-7
    #     values = torch.softmax(values, dim=1)
    #     values = values[:, gt]
    #     values = torch.log(values / (1 - values + eps) + eps)
    # elif selected_dim == "max-log-odds":
    #     eps = 1e-7
    #     values = torch.softmax(values, dim=1)
    #     values = values[:, torch.argmax(values[-1])]
    #     values = torch.log(values / (1 - values + eps) + eps)
    # elif selected_dim == "gt-logistic-odds":
    #     assert "gt" in kwargs.keys()
    #     gt = kwargs["gt"]
    #     assert gt == 0 or gt == 1
    #     eps = 1e-7
    #     assert len(values.shape) == 2 and values.shape[1] == 1
    #     values = torch.sigmoid(values)[:, 0]
    #     if gt == 0:
    #         values = 1 - values
    #     else:
    #         values = values
    #     values = torch.log(values / (1 - values + eps) + eps)
    # elif selected_dim == "logistic-odds":
    #     eps = 1e-7
    #     assert len(values.shape) == 2 and values.shape[1] == 1
    #     values = torch.sigmoid(values)[:, 0]
    #     if torch.round(values[-1]) == 0.:
    #         values = 1 - values
    #     else:
    #         values = values
    #     values = torch.log(values / (1 - values + eps) + eps)
    # elif selected_dim == "gt-prob-log-odds":
    #     assert "gt" in kwargs.keys()
    #     gt = kwargs["gt"]
    #     assert gt == 0 or gt == 1
    #     eps = 1e-7
    #     assert len(values.shape) == 2 and values.shape[1] == 1
    #     values = values[:, 0]
    #     if gt == 0:
    #         values = 1 - values
    #     else:
    #         values = values
    #     values = torch.log(values / (1 - values + eps) + eps)
    # elif selected_dim == "prob-log-odds":
    #     eps = 1e-7
    #     assert len(values.shape) == 2 and values.shape[1] == 1
    #     values = values[:, 0]
    #     if torch.round(values[-1]) == 0.:
    #         values = 1 - values
    #     else:
    #         values = values
    #     values = torch.log(values / (1 - values + eps) + eps)
    # elif selected_dim == None:
    #     values = values
    # else:
    #     raise Exception(f"Unknown [selected_dim] {selected_dim}.")
    CI = []
    # for S in tqdm(masks, desc="calculating CI (one sample)", ncols=100):
    for S in masks:
        subset_mask, subset_indice = generate_subset_masks(set_mask=S, all_masks=masks)
        coef_S = torch.sum(subset_mask, dim=1) + torch.sum(S)
        coef_S = torch.pow(-1, coef_S).float()
        output_S = values[subset_indice]
        CI_S = torch.dot(coef_S, output_S)
        CI.append(CI_S)

    CI = torch.FloatTensor(CI).to(device)
    return masks, CI


def calculate_approx_CI():
    # TODO: the calculation of the precise CI is NP-hard! O(2^d). Need to find an approximation method.
    raise NotImplementedError


def calculate_CI(model, input, baseline, mode="precise", selected_output_dim="max", **kwargs):
    if mode == "precise":
        masks, outputs = calculate_all_subset_outputs(model, input, baseline)
        masks, CI = calculate_precise_CI(masks, outputs, selected_output_dim, **kwargs)
        return masks, CI
    elif mode == "approx":
        calculate_approx_CI()


class ConditionedInteraction(object):
    def __init__(self, mode="precise"):
        self.mode = mode

    def calculate(self, model, input, baseline, selected_output_dim="max", **kwargs):
        return calculate_CI(model, input, baseline, self.mode, selected_output_dim, **kwargs)


def get_all_subset_rewards(model, input, baseline, selected_output_dim, **kwargs):
    masks, outputs = calculate_all_subset_outputs(model, input, baseline)
    rewards = get_reward(outputs, selected_output_dim, **kwargs)
    return masks, rewards

