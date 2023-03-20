import argparse
import os
import os.path as osp
from tools.utils import makedirs, set_seed
import json
import torch
import warnings


def generate_dataset_model_desc(args):
    return "dataset-{}-model-{}-epoch{}-seed{}-bs{}-logspace-{}-lr{}".format(
        args.dataset, args.arch, args.epoch, args.seed, args.batch_size, args.logspace, args.train_lr
    )


def generate_baseline_desc(args):
    return f"baseline_{args.baseline_config}_ci_{args.ci_config}_vfunc_{args.selected_dim}"


def generate_finetune_desc(args):
    if "bound_threshold" not in vars(args).keys() or args.bound_threshold is None:
        return f"finetune_lr_{args.finetune_lr}_max_iter_{args.finetune_max_iter}_suppress_{args.suppress_ratio}"
    else:
        return f"finetune_lr_{args.finetune_lr}_max_iter_{args.finetune_max_iter}_suppress_{args.suppress_ratio}_bound_{args.bound_threshold}"


def generate_aog_remove_noisy_desc(args):
    return f"objective_{args.objective}_maxmerge_{args.max_n_merges}"


def judge_task(args):
    if args.dataset in ["CoLA", "SST-2"] and args.arch in ["lstm2_uni", "cnn"]:
        return "logistic_regression"
    raise NotImplementedError(f"[set_exp.py (judge_task)] Unrecognized task for dataset-DNN {args.dataset}-{args.arch}.")


def makedirs_for_train_model(args):
    if args.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"

    args.dataset_model = generate_dataset_model_desc(args)

    set_seed(args.seed)

    # the model path
    args.model_path = osp.join(args.model_path, args.dataset_model)
    makedirs(args.model_path)

    args.task = judge_task(args)

    with open(osp.join(args.model_path, "hparams.json"), "w") as f:
        json.dump(vars(args), f, indent=4)


def makedirs_for_finetune_baseline_single(args):
    if args.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"

    args.dataset_model = generate_dataset_model_desc(args)
    args.baseline_desc = generate_baseline_desc(args)
    args.finetune_desc = generate_finetune_desc(args)

    args.model_path = osp.join(args.model_path, args.dataset_model, "model.pt")

    set_seed(args.seed)
    args.task = judge_task(args)

    args.save_root = osp.join(args.save_root, args.dataset_model, args.baseline_desc, args.finetune_desc)

    makedirs(args.save_root)

    with open(osp.join(args.save_root, "hparams.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    args.batch_size = 1


def makedirs_for_eval_model(args):
    if args.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"

    args.dataset_model = generate_dataset_model_desc(args)
    args.baseline_desc = generate_baseline_desc(args)
    args.save_root = osp.join(
        args.save_root, args.dataset_model,
        args.baseline_desc
    )
    if args.baseline_config == "custom_single":
        args.finetune_desc = generate_finetune_desc(args)
        args.save_root = osp.join(args.save_root, args.finetune_desc)
        args.baseline_root = osp.join(
            args.baseline_root,
            args.dataset_model,
            f"baseline_pad_ci_{args.ci_config}_vfunc_{args.selected_dim}",
            args.finetune_desc
        )
    makedirs(args.save_root)

    set_seed(args.seed)
    args.model_path = osp.join(args.model_path, args.dataset_model, "model.pt")
    args.task = judge_task(args)

    with open(osp.join(args.save_root, "hparams.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    args.batch_size = 1


def makedirs_for_visualize_AOG_remove_noisy(args):
    if args.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"

    # Load the folder of the saved interaction & baseline
    args.dataset_model = generate_dataset_model_desc(args)
    args.baseline_desc = generate_baseline_desc(args)
    args.interaction_root = osp.join(
        args.interaction_root, args.dataset_model,
        args.baseline_desc
    )
    assert osp.exists(args.interaction_root), args.interaction_root
    if args.baseline_config == "custom_single":
        args.finetune_desc = generate_finetune_desc(args)
        args.interaction_root = osp.join(args.interaction_root, args.finetune_desc)
        args.baseline_root = osp.join(
            "../saved-baselines",
            args.dataset_model,
            f"baseline_pad_ci_{args.ci_config}_vfunc_{args.selected_dim}",
            args.finetune_desc
        )
        assert osp.exists(args.baseline_root)

    # Revise the save folder
    args.aog_desc = generate_aog_remove_noisy_desc(args)
    args.save_root = osp.join(
        args.save_root, args.aog_desc,
        args.dataset_model, args.baseline_desc
    )
    if args.baseline_config == "custom_single":
        args.save_root = osp.join(args.save_root, args.finetune_desc)

    makedirs(args.save_root)

    with open(osp.join(args.save_root, "hparams.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
