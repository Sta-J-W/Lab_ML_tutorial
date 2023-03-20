import argparse
import os
import os.path as osp
from tools.utils import makedirs, set_seed
import json
import torch
# from config import TRAIN_ARGS
import warnings


def generate_dataset_model_desc(args):
    return "dataset-{}-model-{}-epoch{}-seed{}-bs{}-logspace-{}-lr{}".format(
        args.dataset, args.arch, args.epoch, args.model_seed, args.batch_size, args.logspace, args.train_lr
    )


def generate_baseline_desc(args):
    return f"baseline_{args.baseline_config}_ci_{args.ci_config}_vfunc_{args.selected_dim}"


def generate_finetune_desc(args):
    if "bound_threshold" not in vars(args).keys() or args.bound_threshold is None:
        return f"finetune_lr_{args.finetune_lr}_max_iter_{args.finetune_max_iter}_suppress_{args.suppress_ratio}"
    else:
        return f"finetune_lr_{args.finetune_lr}_max_iter_{args.finetune_max_iter}_suppress_{args.suppress_ratio}_bound_{args.bound_threshold}"


def generate_aog_remove_noisy_desc(args):
    return f"objective_{args.objective}_maxmerge_{args.max_n_merges}_thres-approx_{args.thres_approx_error}"



def makedirs_for_train_model(args):
    args.dataset_model = generate_dataset_model_desc(args)

    args.seed = args.model_seed
    set_seed(args.seed)

    # the model path
    args.model_path = osp.join(args.model_path, args.dataset_model)
    makedirs(args.model_path)

    with open(osp.join(args.model_path, "hparams.json"), "w") as f:
        json.dump(vars(args), f, indent=4)


def makedirs_for_finetune_baseline_single(args):
    args.dataset_model = generate_dataset_model_desc(args)
    args.baseline_desc = generate_baseline_desc(args)
    args.finetune_desc = generate_finetune_desc(args)
    save_root = osp.join(
        args.save_root, args.dataset_model,
        args.baseline_desc, args.finetune_desc
    )
    args.save_root = save_root
    makedirs(args.save_root)
    args.model_path = osp.join(args.model_path, args.dataset_model, "model.pt")

    args.seed = args.model_seed
    set_seed(args.seed)

    with open(osp.join(args.save_root, "hparams.json"), "w") as f:
        json.dump(vars(args), f, indent=4)


def makedirs_for_eval_model(args):
    args.dataset_model = generate_dataset_model_desc(args)
    args.baseline_desc = generate_baseline_desc(args)


    args.save_root = osp.join(
        args.save_root, args.dataset_model,
        args.baseline_desc
    )
    if args.baseline_config == "custom_single" or args.baseline_config == "custom_single_together":
        args.save_root = osp.join(args.save_root, generate_finetune_desc(args))
    makedirs(args.save_root)

    args.seed = args.model_seed

    # the model path
    args.model_path = os.path.join(args.model_path, args.dataset_model, "model.pt")

    set_seed(args.seed)

    # load the customized baseline
    args.baseline_root = None
    assert args.baseline_config in ["mean", "zero", "custom_single", "custom_single_together"]
    args.baseline_root = osp.join(
        "../saved-baselines",
        args.dataset_model,
        f"baseline_mean_ci_{args.ci_config}_vfunc_{args.selected_dim}",  # TODO: may cause bug here
        generate_finetune_desc(args)
    )

    with open(osp.join(args.save_root, "hparams.json"), "w") as f:
        json.dump(vars(args), f, indent=4)


def makedirs_for_visualize_AOG_remove_noisy(args):
    # ==============================
    #   1. the dataset and model
    # ==============================
    args.dataset_model = generate_dataset_model_desc(args)

    # ==============================
    #   2. the baseline config
    # ==============================
    args.baseline_desc = generate_baseline_desc(args)

    args.interaction_root = osp.join(
        args.interaction_root, args.dataset_model,
        args.baseline_desc
    )

    args.aog_desc = generate_aog_remove_noisy_desc(args)
    args.save_root = osp.join(
        args.save_root, args.aog_desc,
        args.dataset_model, args.baseline_desc
    )

    # ====================================
    #   3. if there is finetune configs
    # ====================================
    if args.baseline_config == "custom_single" or args.baseline_config == "custom_single_together":
        args.finetune_desc = generate_finetune_desc(args)
        args.interaction_root = osp.join(args.interaction_root, args.finetune_desc)
        args.save_root = osp.join(args.save_root, args.finetune_desc)

    assert osp.exists(args.interaction_root)
    makedirs(args.save_root)

    with open(osp.join(args.save_root, "hparams.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
