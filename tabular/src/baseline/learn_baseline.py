import os
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import sys
sys.path.append("..")
from interaction.interaction_utils import calculate_all_subset_coef_matrix
from interaction import ConditionedInteraction
from tools.utils import makedirs, AverageMeter, save_obj
from .baseline_plot import *


def get_suppressed_IS(CI_mean, all_masks, suppress_ratio):
    '''
    To make I(S) more sparse, we aim to suppress I(S)'s whose absolute value is small. In other words, we
      - first sort the mean of I(S) by their absolute value in an ascending manner;
      - then we choose those S's whose corresponding |I(S)| is small;
      - the loss to train the baseline value can thus be formulated as \sum_S |I(S)|.
    In this function, we are given the mean values of I(S) over all samples, and the corresponding S's (repr.
      by 2^n masks). Then we choose 90% (by default) of |I(S)|'s and suppress them.
    Note: below, n = #of feature dims

    :param CI_mean: mean of I(S) over all samples. Can be np.ndarray or torch.Tensor, with shape (2^n,)
    :param all_masks: masks representing all S's. E.g. mask = (0, 1, 1, 0) -> S = {2, 3}, with shape (2^n, n)
    :param suppress_ratio: suppress some ratio of |I(S)|
    :return:
      1. all_masks (representing all S's), with shape (2^n, n)
      2. target_indices: selected S's, represented by their indices in 'all_masks', with shape (#selected-S,)
      3. sign_list: indicate whether |I(S)| is positive (+1) or negative (-1), with shape (#selected-S,)
    '''
    if not isinstance(CI_mean, torch.Tensor):
        CI_mean = torch.Tensor(CI_mean)
    assert CI_mean.shape[0] == all_masks.shape[0]
    n_masks = all_masks.shape[0]
    n_attributes = all_masks.shape[1]
    target_indices = torch.argsort(torch.abs(CI_mean))
    target_indices = target_indices[:int(n_masks * suppress_ratio)]
    # for I(S) > 0: decrease (+); for I(S) < 0: increase (-)
    sign_list = CI_mean[target_indices] > 0
    sign_list = 2 * sign_list.float() - 1
    return all_masks, target_indices, sign_list


def get_suppressed_vS_coef(all_masks, target_indices, sign_list):
    '''
      Then we can rewrite the loss function (\sum_S |I(S)|) into the weighted-sum of v(S), by removing the absolute
    value sign.
    :param all_masks: see above, with shape (2^n, n)
    :param target_indices: see above, with shape (#selected-S,)
    :param sign_list: see above, with shape (#selected-S,)
    :return: 1. all_masks: (representing all S's), (2^n, n).
             2. coefs: the coefficients before each v(S), with shape (2^n,).
    '''
    n_masks = all_masks.shape[0]
    n_attributes = all_masks.shape[1]
    coefs = calculate_all_subset_coef_matrix(n_attributes)
    coefs = coefs[target_indices]
    sign_list = sign_list.view(-1, 1)
    coefs = coefs * sign_list
    coefs = coefs.sum(dim=0)
    return all_masks, coefs


def learn_baseline_suppress_IS_single(
    model, X, y,
    CI_sample, all_masks, ci_config, selected_dim,
    suppress_ratio, baseline_min, baseline_max, baseline_init,
    baseline_lr, device, max_iter, val_freq, save_root
):
    '''
    This function is the main function for finetuning the baseline values.
    :param model: the target model
    :param X: one data point (with shape [d]) for baseline fintuning
    :param y: the gt-label for the sampled data (with shape [])
    :param CI_sample: the np.ndarray representing mean values of I(S)
    :param all_masks: the boolean masks representing all S's
    :param ci_config: the mode for calculating I(S)
    :param selected_dim: the configuration for v-function, e.g. log-odds of the max-dim.
    :param suppress_ratio:
    :param baseline_min: the lower bound of baseline values
    :param baseline_max: ... upper ...
    :param baseline_init: the initialization of baseline values (need to be the same as in the calculation of I(S))
    :param baseline_lr: the learning rate for finetuning baseline values
    :param device:
    :param max_iter: the maximum iteration for baseline training
    :param val_freq: how many epochs to validate the distribution of I(S)
    :param save_root: the folder for saving the results
    :return: None, but save
      1.
    '''

    if osp.exists(osp.join(save_root, "baseline_final.pth")):
        print("Baseline values were already trained.")
        return

    assert len(X.shape) == 1 and len(y.shape) == 0
    input, label = X.clone(), y.clone()
    input, label = input.to(device), label.to(device)

    baselines = baseline_init.clone().requires_grad_(True)
    baselines = baselines.float()
    baselines.requires_grad_(True)
    lr_list = np.logspace(np.log10(baseline_lr), np.log10(baseline_lr) - 1)

    # setting the model status
    model.requires_grad = False  # dosen't change the gradient for the model.
    model.eval()

    _, target_indices, sign_list = get_suppressed_IS(CI_mean=CI_sample, all_masks=all_masks, suppress_ratio=suppress_ratio)
    _, coefs = get_suppressed_vS_coef(all_masks=all_masks, target_indices=target_indices, sign_list=sign_list)

    train_process_data = {
        "loss": [],
        "CI_sample": [CI_sample.copy()],
        "baselines": [baselines.clone()],
        "v-coef": [coefs.clone()]
    }

    makedirs(osp.join(save_root, "CI_checkpoints"))
    makedirs(osp.join(save_root, "coef_figs"))
    makedirs(osp.join(save_root, "CI_figs"))
    makedirs(osp.join(save_root, "CI_accumulated_figs"))
    makedirs(osp.join(save_root, "CI_trajectory"))

    plot_loss(train_process_data["loss"], osp.join(save_root, "loss.png"))
    plot_coef_distribution(train_process_data["v-coef"][-1], osp.join(save_root, f"coef-init.png"))
    plot_baseline_values(train_process_data["baselines"], osp.join(save_root, f"baselines.png"))
    plot_CI_mean(train_process_data["CI_sample"][-1], osp.join(save_root, f"CI_sample-init.png"))

    for itr in range(max_iter):
        subset_masks = all_masks[coefs != 0]
        subset_masks = torch.BoolTensor(subset_masks).to(device)
        coefs_nonzero = coefs[coefs != 0].to(device)

        loss_meter = AverageMeter()
        # ==========================================
        #    1. calculate grad for this sample
        # ==========================================
        masked_inputs = torch.where(subset_masks, input.expand_as(subset_masks), baselines.expand_as(subset_masks))
        masked_outputs = model(masked_inputs)
        if selected_dim == "max":
            masked_outputs = masked_outputs[:, label]  # v(S)
        elif selected_dim == "max-log-odds":
            eps = 1e-7
            masked_outputs = torch.softmax(masked_outputs, dim=1)
            masked_outputs = masked_outputs[:, label]
            masked_outputs = torch.log(masked_outputs / (1 - masked_outputs + eps) + eps)
        elif selected_dim == "logistic-odds":
            eps = 1e-7
            masked_outputs = torch.sigmoid(masked_outputs)[:, 0]
            if label.item() == 0:
                masked_outputs = 1 - masked_outputs
            elif label.item() == 1:
                pass
            else:
                raise Exception(f"Unknown label: {label}")
            masked_outputs = torch.log(masked_outputs / (1 - masked_outputs + eps) + eps)
        elif selected_dim == "gt-log-odds":
            gt = y.item()
            eps = 1e-7
            masked_outputs = torch.softmax(masked_outputs, dim=1)
            masked_outputs = masked_outputs[:, gt]
            masked_outputs = torch.log(masked_outputs / (1 - masked_outputs + eps) + eps)
        elif selected_dim == "gt-logistic-odds":
            gt = y.item()
            assert gt == 0 or gt == 1
            eps = 1e-7
            assert len(masked_outputs.shape) == 2 and masked_outputs.shape[1] == 1
            masked_outputs = torch.sigmoid(masked_outputs)[:, 0]
            if gt == 0:
                masked_outputs = 1 - masked_outputs
            else:
                masked_outputs = masked_outputs
            masked_outputs = torch.log(masked_outputs / (1 - masked_outputs + eps) + eps)
        elif selected_dim == "gt-prob-log-odds":
            gt = y.item()
            assert gt == 0 or gt == 1
            eps = 1e-7
            assert len(masked_outputs.shape) == 2 and masked_outputs.shape[1] == 1
            masked_outputs = masked_outputs[:, 0]
            if gt == 0:
                masked_outputs = 1 - masked_outputs
            else:
                masked_outputs = masked_outputs
            masked_outputs = torch.log(masked_outputs / (1 - masked_outputs + eps) + eps)
        elif selected_dim == "0":
            masked_outputs = masked_outputs[:, 0]
        else:
            raise NotImplementedError(f"{selected_dim} not implemented")

        loss = torch.dot(coefs_nonzero, masked_outputs)
        loss_meter.update(loss.item())
        grad = torch.autograd.grad(loss, baselines)[0]

        # ==========================================
        #    2. update & save the baseline values
        # ==========================================
        baselines.data = baselines.data - lr_list[itr] * grad
        # baselines.data = baselines.data - baseline_lr * grad
        baselines = torch.max(torch.min(baselines, baseline_max), baseline_min)\
                         .clone().detach().requires_grad_(True).float()  # clamp
        train_process_data["loss"].append(loss_meter.avg)
        train_process_data["baselines"].append(baselines.clone())

        if itr % val_freq == 0:
            itr_save_folder = osp.join(save_root, "CI_checkpoints", f"iter_{itr}")
            makedirs(itr_save_folder)
            # ==========================================
            #    3. re-calculate I(S), update loss coef
            # ==========================================
            print(f" -> Validate I(S) for iteration {itr}.")
            interaction_calculator = ConditionedInteraction(mode=ci_config)

            masks, CI = interaction_calculator.calculate(
                model, input, baselines,
                selected_output_dim=selected_dim,
                gt=y.item()
            )

            CI = CI.cpu().numpy()
            masks = masks.cpu().numpy()
            np.save(osp.join(itr_save_folder, "CI.npy"), CI)
            np.save(osp.join(itr_save_folder, "masks.npy"), masks)

            CI_sample = CI.copy()

            _, target_indices, sign_list = get_suppressed_IS(CI_mean=CI_sample, all_masks=all_masks, suppress_ratio=suppress_ratio)
            _, coefs = get_suppressed_vS_coef(all_masks=all_masks, target_indices=target_indices, sign_list=sign_list)

            train_process_data["CI_sample"].append(CI_sample.copy())
            train_process_data["v-coef"].append(coefs.clone())

            # ==========================================
            #    4. do some visualization
            # ==========================================
            plot_loss(train_process_data["loss"], osp.join(save_root, "loss.png"))
            plot_coef_distribution(train_process_data["v-coef"][-1], osp.join(save_root, "coef_figs", f"coef-iter-{itr}.png"))
            plot_baseline_values(train_process_data["baselines"], osp.join(save_root, f"baselines.png"))
            plot_CI_mean(train_process_data["CI_sample"][-1], osp.join(save_root, "CI_figs", f"CI_sample-iter-{itr}.png"))
            plot_CI_mean(train_process_data["CI_sample"], osp.join(save_root, "CI_accumulated_figs", f"CI_sample-accumulated-iter-{itr}-f.png"), order_cfg="first")
            plot_CI_mean(train_process_data["CI_sample"], osp.join(save_root, "CI_accumulated_figs", f"CI_sample-accumulated-iter-{itr}-d.png"), order_cfg="descending")



    # CI_sample_trajectory = np.stack(train_process_data["CI_sample"])
    # init_ci_order = np.argsort(-CI_sample_trajectory[0])
    # for i, s_id in enumerate(init_ci_order):
    #     plot_ci_trajectory(
    #         ci_trajectory=CI_sample_trajectory[:, s_id],
    #         save_path=osp.join(save_root, "CI_trajectory", f"descending_{str(i).zfill(5)}_sid_{str(s_id).zfill(5)}.png")
    #     )

    # ==========================================
    #    save plotted data during training process
    # ==========================================
    for k in train_process_data.keys():
        if isinstance(train_process_data[k][0], int) or isinstance(train_process_data[k][0], float):
            save_obj(train_process_data[k], osp.join(save_root, f"{k}.bin"))
        elif isinstance(train_process_data[k][0], torch.Tensor):
            train_process_data[k] = torch.stack(train_process_data[k]).cpu().detach()
            torch.save(train_process_data[k], osp.join(save_root, f"{k}.pth"))
        elif isinstance(train_process_data[k][0], np.ndarray):
            train_process_data[k] = np.stack(train_process_data[k])
            np.save(osp.join(save_root, f"{k}.npy"), train_process_data[k])
        else:
            raise NotImplementedError(f"Unknown type: {type(train_process_data[k][0])}")

    # save the baselines at the final iteration
    torch.save(baselines.detach().cpu(), osp.join(save_root, "baseline_final.pth"))
