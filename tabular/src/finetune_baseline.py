import torch
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import argparse
import dataset
from baseline import learn_baseline_suppress_IS_single, mean_baseline
from tools.train import train_model
from tools.train import load_model
from tools.utils import makedirs
from set_exp import makedirs_for_finetune_baseline_single
from interaction import ConditionedInteraction


print("-------Parsing program arguments--------")
parser = argparse.ArgumentParser(description="Finetune the baseline values")
## the basic setting of exp
parser.add_argument('--device', default=2, type=int,
                    help="set the device.")
parser.add_argument("--dataset", default="commercial", type=str,
                    help="set the dataset used.")
parser.add_argument("--arch", default="mlp5", type=str,
                    help="the network architecture.")
# set the path for data
parser.add_argument('--data_path', default='/data1/limingjie/data/tabular', type=str,
                    help="path for dataset.")
# path for saved file
parser.add_argument("--save_root", default="../saved-baselines", type=str,
                    help='the path of saved fig.')
# set the (pre-trained) model path.
parser.add_argument("--model_path", default="../saved-models", type=str,
                    help='the path of pretrained model.')

## the setting for the pre-trained model
# set the model seed
parser.add_argument("--model_seed", default=0, type=int,
                    help="set the seed used for training model.")
# set the batch size for training
parser.add_argument('--batch_size', default=512, type=int,
                    help="set the batch size for training.")
# set the learning rate for training
parser.add_argument('--train_lr', default=0.01, type=float,
                    help="set the learning rate for training.")
# set the decay of learning rate
parser.add_argument("--logspace", default=1, type=int,
                    help='the decay of learning rate.')
# set the number of epochs for training model.
parser.add_argument("--epoch", default=300, type=int,
                    help='the number of iterations for training model.')


## the settings for calculating $CI(S)$
parser.add_argument("--selected-dim", default="gt-log-odds",
                    help="The value function, can be selected from: (None) | max | gt | gt-log-odds | max-log-odds | logistic-odds")
parser.add_argument("--baseline-config", default="mean",
                    help="The config of the baseline value")
parser.add_argument("--ci-config", default="precise",
                    help="use the precise/approximated CI (multi-variate interaction)")

## the settings for finetuning the baselines
parser.add_argument("--bound-threshold", type=float, default=0.1,
                    help="bound of baseline value: [_-th*std, _+th*std]")
parser.add_argument("--finetune-lr", type=float, default=0.000001,
                    help="the learning rate to optimize baseline value")
parser.add_argument("--suppress-ratio", type=float, default=1.0,
                    help="set the loss to suppress r% interactions. "
                         "particularly, if set as 1, the loss is L1-loss.")
parser.add_argument("--finetune-max-iter", type=int, default=20,
                    help="how many iterations to optimize baseline value")
parser.add_argument("--finetune-val-freq", type=int, default=1,
                    help="how many epochs to validate the distribution of I(S)")
parser.add_argument("--eval-sample-num", type=int, default=1,
                    help="evaluate I(S) for x samples.")

args = parser.parse_args()
makedirs_for_finetune_baseline_single(args)

print("-----------preparing dataset-----------")
print("dataset - {}".format(args.dataset))
if args.dataset in ["census", "commercial", "bike"]:
    dataset_info, \
    X_train, y_train, X_test, y_test, \
    X_train_sampled, y_train_sampled, \
    X_test_sampled, y_test_sampled, \
    train_loader = dataset.load_tabular(args)
else:
    raise Exception(f"Unknown dataset: {args.dataset}")

if "logistic" in args.arch:
    task = {
        "census": "logistic_regression",
        "commercial": "logistic_regression",
        "bike": "regression"
    }[args.dataset]
else:
    task = {
        "census": "classification",
        "commercial": "classification",
        "bike": "regression"
    }[args.dataset]


# ===============================================
#   train the model first (if exist, then load)
# ===============================================
print("------------preparing model------------")
model = load_model(args, args.model_path, X_train, y_train, X_test, y_test)
# model = train_model(args, X_train, y_train, X_test, y_test, train_loader, task)
model.eval()

if args.baseline_config == "mean":
    baseline = mean_baseline(X_train).to(args.device)
else:
    raise Exception(f"Unknown baseline config: {args.baseline_config}")


# ===============================================
#   set the configs, initialize the calculator
# ===============================================
print("------------evaluating CI & finetuning baseline------------")
if args.baseline_config == "mean":
    baseline = mean_baseline(X_train).to(args.device)
else:
    raise Exception(f"Unknown baseline config: {args.baseline_config}")

if args.ci_config not in ["precise"]:
    raise Exception(f"Unknown CI config: {args.ci_config}")
interaction_calculator = ConditionedInteraction(mode=args.ci_config)

# ===============================================
#   the sampled data, for calculating I(S)
# ===============================================
if args.eval_sample_num is None:
    sampled_data = {
        "train": [X_train_sampled, y_train_sampled],
        "test": [X_test_sampled, y_test_sampled]
    }
else:
    n_sample = args.eval_sample_num
    sampled_data = {
        "train": [X_train_sampled[:n_sample], y_train_sampled[:n_sample]],
        "test": [X_test_sampled[:n_sample], y_test_sampled[:n_sample]]
    }

# ====================================================
#     the min and max of baselines
# ====================================================
bound_threshold = args.bound_threshold
baseline_min = X_train.mean(dim=0) - bound_threshold * X_train.std(dim=0)  # (1, num_of_features)
baseline_max = X_train.mean(dim=0) + bound_threshold * X_train.std(dim=0)  # (1, num_of_features)


# ====================================================
#   start calculating CI(S) and finetuning baselines
# ====================================================
for split in ["train", "test"]:
    X_sampled, y_sampled = sampled_data[split]

    for idx in tqdm(range(X_sampled.shape[0]), desc=f"Finetuning baselines ({split} set)"):
        input, target = X_sampled[idx], y_sampled[idx]

        masks, CI = interaction_calculator.calculate(
            model, input, baseline,
            selected_output_dim=args.selected_dim,
            gt=target.item()
        )

        CI = CI.cpu().numpy()
        masks = masks.cpu().numpy()
        save_folder = osp.join(args.save_root, f"{split}/class-{target.item()}/sample-{str(idx).zfill(4)}")
        makedirs(save_folder)
        print("save at:", save_folder)
        np.save(osp.join(save_folder, "CI-init.npy"), CI)

        with open(osp.join(save_folder, "net-output.txt"), "w") as f:
            f.write(f"sum of I(S): {CI.sum()}\n")
            f.write(f"v(S): {model(input.unsqueeze(0))}\n")
            f.write(f"y: {target.item()}\n")

        learn_baseline_suppress_IS_single(
            model=model, X=input, y=target, suppress_ratio=args.suppress_ratio,
            baseline_min=baseline_min, baseline_max=baseline_max,
            CI_sample=CI, all_masks=masks, baseline_init=baseline,
            baseline_lr=args.finetune_lr, ci_config=args.ci_config,
            selected_dim=args.selected_dim, device=args.device,
            max_iter=args.finetune_max_iter, val_freq=args.finetune_val_freq,
            save_root=save_folder
        )