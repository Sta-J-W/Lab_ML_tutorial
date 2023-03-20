import argparse
import inspect
import os
import os.path as osp
from tools.utils import makedirs, set_seed
import json
import socket
import torch
import warnings
import re
from typing import List, Tuple, Dict


def _json_encoder_default(obj):
    if isinstance(obj, torch.Tensor):
        return str(obj)
    elif isinstance(obj, type(lambda x: x)):
        return inspect.getsource(obj).strip()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_args(args, save_path):
    with open(save_path, "w") as f:
        json.dump(vars(args), f, indent=4,
                  default=_json_encoder_default)


def generate_dataset_model_desc(args):
    return f"dataset={args.dataset}" \
           f"{'_balance' if 'balance' in vars(args) and args.balance else ''}" \
           f"_model={args.arch}" \
           f"_epoch={args.n_epoch}" \
           f"_bs={args.batch_size}" \
           f"_lr={args.lr}" \
           f"_logspace={args.logspace}" \
           f"_seed={args.seed}"


def generate_adv_train_desc(args):
    return f"step-size={args.adv_step_size}" \
           f"_epsilon={args.adv_epsilon}" \
           f"_n-step={args.adv_n_step}"


def parse_dataset_model_desc(dataset_model_desc: str) -> Dict:
    """
    Parse model args
    :param dataset_model_desc: the arg string
    :return: dict

    >>> parse_dataset_model_desc("dataset=census_balance_model=mlp5_epoch=5000_bs=512_lr=0.1_logspace=2_seed=0")
    {'dataset': 'census', 'arch': 'mlp5', 'balance': True, 'n_epoch': 5000, 'batch_size': 512, 'lr': 0.1, 'logspace': 2, 'seed': 0}

    >>> parse_dataset_model_desc("dataset=commercial_model=mlp5_epoch=5000_bs=512_lr=0.01_logspace=2_seed=0")
    {'dataset': 'commercial', 'arch': 'mlp5', 'balance': False, 'n_epoch': 5000, 'batch_size': 512, 'lr': 0.01, 'logspace': 2, 'seed': 0}

    >>> parse_dataset_model_desc("dataset=gaussian_rule_001_regression_10d_v1_model=mlp5_sigmoid_epoch=500_bs=512_lr=0.01_logspace=1_seed=0")
    {'dataset': 'gaussian_rule_001_regression_10d_v1', 'arch': 'mlp5_sigmoid', 'balance': False, 'n_epoch': 500, 'batch_size': 512, 'lr': 0.01, 'logspace': 1, 'seed': 0}

    >>> parse_dataset_model_desc("dataset=gaussian_rule_001_regression_10d_v1_model=mlp5_sigmoid_epoch=100_bs=512_lr=0.01_logspace=1_seed=0_step-size=0.01_epsilon=0.1_n-step=20")
    {'dataset': 'gaussian_rule_001_regression_10d_v1', 'arch': 'mlp5_sigmoid', 'balance': False, 'n_epoch': 100, 'batch_size': 512, 'lr': 0.01, 'logspace': 1, 'seed': 0, 'adv_step_size': 0.01, 'adv_epsilon': 0.1, 'adv_n_step': 20}
    """
    pattern = r"dataset=(.+)" \
              r"\_model=(.+)" \
              r"\_epoch=(.+)" \
              r"\_bs=(.+)" \
              r"\_lr=(.+)" \
              r"\_logspace=(.+)" \
              r"\_seed=(.+)" \
              r"\_step-size=(.+)" \
              r"\_epsilon=(.+)" \
              r"\_n-step=(.+)"
    match = re.match(pattern, dataset_model_desc)

    if match is not None:
        dataset, arch, n_epoch, batch_size, lr, logspace, seed, adv_step_size, adv_epsilon, adv_n_step = match.groups()
        balance = "_balance" in dataset
        if balance:
            dataset = "_".join(dataset.split("_")[:-1])
        n_epoch = int(n_epoch)
        batch_size = int(batch_size)
        lr = float(lr)
        logspace = int(logspace)
        seed = int(seed)
        adv_step_size = float(adv_step_size)
        adv_epsilon = float(adv_epsilon)
        adv_n_step = int(adv_n_step)

        return {
            "dataset": dataset,
            "arch": arch,
            "balance": balance,
            "n_epoch": n_epoch,
            "batch_size": batch_size,
            "lr": lr,
            "logspace": logspace,
            "seed": seed,
            "adv_step_size": adv_step_size,
            "adv_epsilon": adv_epsilon,
            "adv_n_step": adv_n_step
        }
    else:
        pattern = r"dataset=(.+)" \
                  r"\_model=(.+)" \
                  r"\_epoch=(.+)" \
                  r"\_bs=(.+)" \
                  r"\_lr=(.+)" \
                  r"\_logspace=(.+)" \
                  r"\_seed=(.+)"
        match = re.match(pattern, dataset_model_desc)
        assert match is not None

        dataset, arch, n_epoch, batch_size, lr, logspace, seed = match.groups()
        balance = "_balance" in dataset
        if balance:
            dataset = "_".join(dataset.split("_")[:-1])
        n_epoch = int(n_epoch)
        batch_size = int(batch_size)
        lr = float(lr)
        logspace = int(logspace)
        seed = int(seed)  # TODO: add support for adversarial training

        return {
            "dataset": dataset,
            "arch": arch,
            "balance": balance,
            "n_epoch": n_epoch,
            "batch_size": batch_size,
            "lr": lr,
            "logspace": logspace,
            "seed": seed,
        }


def _init_model_setting(args):
    if args.dataset in ["mnist", "simplemnist"] and args.arch in ["resnet20", "resnet32", "resnet44", "vgg13_bn", "vgg16_bn"]:
        args.model_kwargs = {"input_channel": 1, "num_classes": 10}
        args.task = "classification"
    elif args.dataset in ["mnist", "simplemnist"] and args.arch in ["lenet"]:
        args.model_kwargs = {}
        args.task = "classification"
    else:
        raise NotImplementedError(f"[Undefined] Dataset: {args.dataset}, Model: {args.arch}")


def setup_finetune_baseline(args):
    assert args.model_args is not None
    args.dataset_model = parse_dataset_model_desc(args.model_args)
    args.dataset = args.dataset_model["dataset"]
    args.arch = args.dataset_model["arch"]
    args.seed = args.dataset_model["seed"]
    args.batch_size = args.dataset_model["batch_size"]
    _init_model_setting(args)
    set_seed(args.seed)

    args.save_root = osp.join(args.save_root, args.model_args, f"manual_segment_{args.segment_version}",
                              f"dim={args.selected_dim}_baseline-init={args.baseline_init}"
                              f"_lb={args.baseline_lb}_ub={args.baseline_ub}"
                              f"_lr={args.finetune_lr}_niter={args.finetune_max_iter}")

    args.manual_segment_root = osp.join(args.manual_segment_root, f"{args.dataset}-segments", args.segment_version)

    makedirs(args.save_root)
    args.hostname = socket.gethostname()
    save_args(args=args, save_path=osp.join(args.save_root, "hparams.json"))