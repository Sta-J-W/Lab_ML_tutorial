import os
import os.path as osp
import logging
import argparse
import torch
import torch.nn as nn
import model
from tools.train import load_model
from tools.utils import get_unmasked_attribute_name, makedirs, save_obj
from tools.remove_noisy import remove_noisy_greedy
from tools.plot import compare_simple_line_chart, visualize_all_CI_descending
import dataset
from baseline import mean_baseline, zero_baseline
from interaction import get_all_subset_rewards
from interaction.convert_matrix import get_reward2harsanyi_mat, get_harsanyi2reward_mat
from set_exp import makedirs_for_eval_model
from interaction import ConditionedInteraction
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Evaluate interaction in deep models")
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
parser.add_argument("--save_root", default="../saved-interactions", type=str,
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
                    help="The value function, can be selected from: gt-log-odds | gt-logistic-odds | gt-prob-log-odds | 0")
parser.add_argument("--baseline-config", default="custom_single",
                    help="The config of the baseline value: custom_single")
parser.add_argument("--model_seeds", default=None, type=str,
                    help="models trained with these seeds share the same baseline (splitted by '-'), e.g. 0-1-2.")
parser.add_argument("--ci-config", default="precise",
                    help="use the precise/approximated CI (multi-variate interaction)")
parser.add_argument("--eval-sample-num", type=int, default=1,
                    help="evaluate I(S) for ? samples.")

## hparams for customized baseline values
parser.add_argument("--bound-threshold", type=float, default=0.1, help="bound of baseline value: [_-th*std, _+th*std]")
parser.add_argument("--finetune-lr", type=float, default=0.000001)
parser.add_argument("--suppress-ratio", type=float, default=1.0)
parser.add_argument("--finetune-max-iter", type=int, default=20)

parser.add_argument("--thres-approx-error", type=float, default=0.05,
                    help="a hyper-parameter for the construction of Omega, i.e. "
                         "sqrt[sum of [v(S)-g(S)]**2] should be less than eps * |v|_2")

args = parser.parse_args()
args.save_root += f"-{args.thres_approx_error}"
makedirs_for_eval_model(args)


# ===============================================
#   prepare the dataset (for train & eval)
# ===============================================
print("-----------preparing dataset-----------")
print("dataset - {}".format(args.dataset))

dataset_info,\
X_train, y_train, X_test, y_test,\
X_train_sampled, y_train_sampled,\
X_test_sampled, y_test_sampled,\
train_loader = dataset.load_tabular(args)



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
        "bike": "regression",
    }[args.dataset]

# ===============================================
#   load the model
# ===============================================
print("------------preparing model------------")
model = load_model(args, args.model_path, X_train, y_train, X_test, y_test)
model.eval()

# ===============================================
#   set the configs, initialize the calculator
# ===============================================
print("------------evaluating CI------------")
if args.baseline_config == "mean":
    baseline = mean_baseline(X_train).to(args.device)
elif args.baseline_config == "zero":
    baseline = zero_baseline(X_train).to(args.device)
elif args.baseline_config == "custom_single" or args.baseline_config == "custom_single_together":
    baseline = None
else:
    raise Exception(f"Unknown baseline config: {args.baseline_config}")

if args.ci_config not in ["precise"]:
    raise Exception(f"Unknown CI config: {args.ci_config}")
# interaction_calculator = ConditionedInteraction(mode=args.ci_config)


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


# ===============================================
#   start calculating CI(S)
# ===============================================
for split in ["train", "test"]:
    X_sampled, y_sampled = sampled_data[split]

    for idx in tqdm(range(X_sampled.shape[0]), desc=f"Calculating CI ({split} set)"):
        input, target = X_sampled[idx], y_sampled[idx]

        split_category_sample = (split, f"class-{target.item()}", f"sample-{str(idx).zfill(4)}")

        baseline = torch.load(osp.join(args.baseline_root, *split_category_sample, "baseline_final.pth"))
        baseline = baseline.to(args.device)


        save_folder = osp.join(args.save_root, *split_category_sample)
        makedirs(save_folder)

        # ==========================================
        #     Begin calculating the Harsanyi
        # ==========================================

        # _, CI_ori = interaction_calculator.calculate(
        #     model, input, baseline,
        #     selected_output_dim=args.selected_dim,
        #     gt=target.item()
        # )

        masks, rewards = get_all_subset_rewards(
            model, input, baseline,
            selected_output_dim=args.selected_dim,
            gt=target.item()
        )

        reward2harsanyi = get_reward2harsanyi_mat(all_masks=masks).to(args.device)
        harsanyi2reward = get_harsanyi2reward_mat(all_masks=masks).to(args.device)
        CI = torch.matmul(reward2harsanyi, rewards)

        _, verbose_greedy = remove_noisy_greedy(rewards, CI.clone(), masks, harsanyi2reward, min_patterns=15, n_greedy=40,
                                                thres_approx_error=args.thres_approx_error, thres_explain_ratio=0.95)
        # _, verbose_lsf = remove_noisy_low_strength_first(rewards, CI.clone(), masks, harsanyi2reward, min_patterns=15)
        _, verbose_lsf = remove_noisy_greedy(rewards, CI.clone(), masks, harsanyi2reward, min_patterns=15, n_greedy=1,
                                             thres_approx_error=args.thres_approx_error, thres_explain_ratio=0.95)

        compare_simple_line_chart(data_series=[verbose_greedy["errors"], verbose_lsf["errors"]],
                                  legends=["greedy to min faith(I) [g=40]", "remove lowest-strength I(S)"],
                                  xlabel="# interaction effects removed", ylabel="faith(I)",
                                  title="faith(I)", save_folder=save_folder, save_name="remove_noisy_error")
        compare_simple_line_chart(data_series=[verbose_greedy["explain_ratios"], verbose_lsf["explain_ratios"],],
                                  legends=["greedy to min faith(I) [g=40]", "remove lowest-strength I(S)"],
                                  xlabel="# interaction effects removed", ylabel="explain ratio",
                                  title="explain ratio", save_folder=save_folder, save_name="remove_noisy_explain_ratio")

        CI = CI.cpu().numpy()
        masks = masks.cpu().numpy()

        # ==========================================
        #     Save the calculated results
        # ==========================================

        with open(osp.join(save_folder, "net-output.txt"), "w") as f:
            f.write(f"sum of I(S): {CI.sum()}\n")
            with torch.no_grad(): output = model(input[None, ...])
            f.write(f"output: {output}\n")
            f.write(f"y: {target.item()}\n\n")

        with open(osp.join(save_folder, "interactions.txt"), "w") as f:
            CI_order = np.argsort(-np.abs(CI))
            for i in range(CI.shape[0]):
                f.write(str(masks[CI_order][i]) + "\t")
                f.write(str(CI[CI_order][i]) + "\n")

        with open(osp.join(save_folder, "remove_noisy.txt"), "w") as f:
            f.write("final result after removing noisy:\n")
            f.write(f"\t# coalitions: {len(verbose_greedy['final_retained'])}\n")
            f.write(f"\tapprox. error: {verbose_greedy['final_error']}\n")
            f.write(f"\texplain ratio: {verbose_greedy['final_ratio']}\n")
            f.write("interactions:\n")
            for i in range(len(verbose_greedy['final_retained'])):
                f.write(str(masks[verbose_greedy['final_retained'][i]]) + "\t")
                f.write(str(CI[verbose_greedy['final_retained'][i]]) + "\n")

        with open(osp.join(save_folder, "remove_noisy_lsf.txt"), "w") as f:
            f.write("final result (low strength first):\n")
            f.write(f"\t# coalitions: {len(verbose_lsf['final_retained'])}\n")
            f.write(f"\tapprox. error: {verbose_lsf['final_error']}\n")
            f.write(f"\texplain ratio: {verbose_lsf['final_ratio']}\n")
            f.write("interactions:\n")
            for i in range(len(verbose_lsf['final_retained'])):
                f.write(str(masks[verbose_lsf['final_retained'][i]]) + "\t")
                f.write(str(CI[verbose_lsf['final_retained'][i]]) + "\n")


        np.save(osp.join(save_folder, "CI.npy"), CI)
        np.save(osp.join(save_folder, "masks.npy"), masks)
        np.save(osp.join(save_folder, "input.npy"), input.squeeze().cpu().numpy())
        np.save(osp.join(save_folder, "baseline.npy"), baseline.squeeze().cpu().numpy())
        save_obj(verbose_greedy["final_retained"], osp.join(save_folder, "final_retained.bin"))
        visualize_all_CI_descending(CI, osp.join(save_folder, "CI_descending.png"))
