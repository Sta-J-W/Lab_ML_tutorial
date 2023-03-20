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
from tools.utils import makedirs, AverageMeter, save_obj, load_obj
from .baseline_plot import *
from tqdm import tqdm






def plot_learn_baseline_suppress_IS_single(save_root):
    '''
    This function ......
    :param save_root: the folder for saving the results
    :return: None, but save
      1.
    '''

    train_process_data = {}

    train_process_data["loss"] = load_obj(osp.join(save_root, "loss.bin"))
    train_process_data["v-coef"] = torch.load(osp.join(save_root, "v-coef.pth"))
    train_process_data["baselines"] = torch.load(osp.join(save_root, "baselines.pth"))
    train_process_data["CI_sample"] = np.load(osp.join(save_root, "CI_sample.npy"))

    max_iter = len(train_process_data["loss"])
    val_freq = max_iter // (len(train_process_data["CI_sample"]) - 1)

    makedirs(osp.join(save_root, "CI_trajectory"))

    CI_sample_trajectory = train_process_data["CI_sample"]
    init_ci_order = np.argsort(-CI_sample_trajectory[0])
    for i, s_id in enumerate(tqdm(init_ci_order, desc="Plotting")):
        plot_ci_trajectory(
            ci_trajectory=CI_sample_trajectory[:, s_id],
            save_path=osp.join(save_root, "CI_trajectory", f"descending_{str(i).zfill(5)}_sid_{str(s_id).zfill(5)}.png")
        )
