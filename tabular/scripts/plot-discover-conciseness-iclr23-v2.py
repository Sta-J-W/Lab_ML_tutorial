import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import json

import sys
sys.path.append(osp.join(osp.dirname(__file__), "../.."))

from InteractionAOG.src.tools.utils import load_obj


def plot_strength_histogram(interactions, n_bins, save_folder, save_name, save_type="png"):
    os.makedirs(save_folder, exist_ok=True)
    plt.figure()
    plt.xlabel(r"relative strength of causal effects "
               r"$|w_{\mathcal{S}}|\ / \ {\max}_{\mathcal{S}'}|w_{\mathcal{S}'}|$")
    plt.ylabel(r"frequency of causal patterns $\mathcal{S}$")
    plt.hist(np.abs(interactions), histtype="stepfilled", bins=n_bins,
             alpha=1.0, weights=np.ones_like(interactions) / interactions.shape[0])
    plt.tight_layout()
    plt.savefig(osp.join(save_folder, f"{save_name}.{save_type}"))
    plt.close("all")


def plot_strength_curve(interactions, save_folder, save_name, save_type="png"):
    plt.figure()
    plt.xlabel(r"causal patterns $\mathcal{S}$ (with strength descending)")
    if len(interactions.shape) == 1:
        plt.ylabel(r"strength of causal effects $|w_{\mathcal{S}}|$")
        strength = np.abs(interactions)
        strength = strength[np.argsort(-strength)]
        plt.plot(np.arange(strength.shape[0]), strength, alpha=1.0)
    else:
        plt.ylabel(r"relative strength of causal effects "
                   r"$|w_{\mathcal{S}}|\ / \ {\max}_{\mathcal{S}'}|w_{\mathcal{S}'}|$")
        for i in range(interactions.shape[0]):
            strength = np.abs(interactions[i])
            strength = strength[np.argsort(-strength)]
            plt.plot(np.arange(strength.shape[0]), strength, alpha=0.2, color="#1f77b4")
    plt.tight_layout()
    plt.savefig(osp.join(save_folder, f"{save_name}.{save_type}"))
    plt.close("all")


def get_config():
    # ======================================================================================
    #      You can specify the dataset & model to plot
    # ======================================================================================
    dataset_model = "dataset-census-model-mlp5-epoch500-seed0-bs512-logspace-1-lr0.01"
    baseline_config = "baseline_custom_single_ci_precise_vfunc_gt-log-odds"
    finetune_config = "finetune_lr_1e-06_max_iter_50_suppress_1.0_bound_0.1"

    # dataset_model = "dataset-census-model-resmlp5-epoch500-seed0-bs512-logspace-1-lr0.01"
    # baseline_config = "baseline_custom_single_ci_precise_vfunc_gt-log-odds"
    # finetune_config = "finetune_lr_1e-06_max_iter_50_suppress_1.0_bound_0.1"

    # dataset_model = "dataset-census-model-mlp2_logistic-epoch500-seed0-bs512-logspace-1-lr0.01"
    # baseline_config = "baseline_custom_single_ci_precise_vfunc_gt-logistic-odds"
    # finetune_config = "finetune_lr_0.0001_max_iter_50_suppress_1.0_bound_0.1"


    # dataset_model = "dataset-commercial-model-mlp5-epoch300-seed0-bs512-logspace-1-lr0.01"
    # baseline_config = "baseline_custom_single_ci_precise_vfunc_gt-log-odds"
    # finetune_config = "finetune_lr_1e-06_max_iter_50_suppress_1.0_bound_0.1"

    # dataset_model = "dataset-commercial-model-resmlp5-epoch300-seed0-bs512-logspace-1-lr0.01"
    # baseline_config = "baseline_custom_single_ci_precise_vfunc_gt-log-odds"
    # finetune_config = "finetune_lr_1e-06_max_iter_50_suppress_1.0_bound_0.1"

    # dataset_model = "dataset-commercial-model-mlp2_logistic-epoch300-seed0-bs512-logspace-1-lr0.01"
    # baseline_config = "baseline_custom_single_ci_precise_vfunc_gt-logistic-odds"
    # finetune_config = "finetune_lr_0.0001_max_iter_50_suppress_1.0_bound_0.1"


    # dataset_model = "dataset-bike-model-mlp5-epoch1000-seed0-bs512-logspace-1-lr0.01"
    # baseline_config = "baseline_custom_single_ci_precise_vfunc_0"
    # finetune_config = "finetune_lr_1e-06_max_iter_50_suppress_1.0_bound_0.1"

    # dataset_model = "dataset-bike-model-resmlp5-epoch1000-seed0-bs512-logspace-1-lr0.01"
    # baseline_config = "baseline_custom_single_ci_precise_vfunc_0"
    # finetune_config = "finetune_lr_1e-06_max_iter_50_suppress_1.0_bound_0.1"

    # ======================================================================================
    return dataset_model, baseline_config, finetune_config


def traverse(root):
    frontier = [root]
    result_folders = []
    while len(frontier) > 0:
        folder = frontier.pop()
        if not osp.isdir(folder):
            continue
        subfolders = os.listdir(folder)
        for subfolder in subfolders:
            if subfolder.startswith("sample"):
                result_folders.append(osp.join(folder, subfolder))
            else:
                frontier.append(osp.join(folder, subfolder))
    return result_folders


def get_save_name(folder):
    dirs = folder.split("/")
    if "train" in dirs:
        save_name = "train"
    else:
        save_name = "test"
    save_name += f"_{dirs[-1]}"
    return save_name


if __name__ == '__main__':

    dataset_model, baseline_config, finetune_config = get_config()

    save_type = "pdf"
    src_root = f"../saved-CI/{dataset_model}/{baseline_config}/{finetune_config}"
    save_root = f"./discover-conciseness-v2/{dataset_model}/{baseline_config}/{finetune_config}"

    sample_folders = traverse(src_root)


    # # ==========================
    # #         直方图
    # # ==========================
    # all_interactions = []
    # # 单个样本
    # for sample_folder in sample_folders:
    #     interactions = np.load(osp.join(sample_folder, "CI.npy"))
    #     interactions = interactions / np.max(np.abs(interactions))
    #     save_name = get_save_name(sample_folder)
    #     plot_strength_histogram(
    #         interactions, n_bins=100,
    #         save_folder=save_root,
    #         save_name=save_name,
    #         save_type=save_type
    #     )
    #     all_interactions.append(interactions)
    #
    # # 总体（归一化后）
    # all_interactions = np.concatenate(all_interactions)
    # plot_strength_histogram(
    #     all_interactions, n_bins=100,
    #     save_folder=save_root,
    #     save_name="all_samples",
    #     save_type=save_type
    # )

    # ==========================
    #      曲线图（降序）
    # ==========================
    all_interactions = []
    # 单个样本
    for sample_folder in sample_folders:
        interactions = np.load(osp.join(sample_folder, "CI.npy"))
        save_name = get_save_name(sample_folder)
        plot_strength_curve(
            interactions,
            save_folder=save_root,
            save_name=f"{save_name}_curve",
            save_type=save_type
        )
        interactions = interactions / np.max(np.abs(interactions))
        all_interactions.append(interactions)

    # 总体（归一化后）
    all_interactions = np.stack(all_interactions)
    plot_strength_curve(
        all_interactions,
        save_folder=save_root,
        save_name="all_samples_curve",
        save_type=save_type
    )

