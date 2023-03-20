import os
import os.path as osp
import logging
import argparse
import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm
from tools.utils import makedirs, save_obj
from tools.train import eval_model
from interaction import ConditionedInteraction
from interaction import get_all_subset_rewards
from interaction.convert_matrix import get_reward2harsanyi_mat, get_harsanyi2reward_mat
from set_exp import makedirs_for_eval_model
from model import load_nlp_model
from dataset import load_nlp_dataset
from baseline import unk_baseline, pad_baseline
from tools.eval_ci import visualize_all_CI_descending
from tools.remove_noisy import remove_noisy_greedy, remove_noisy_low_strength_first
from tools.plot import compare_simple_line_chart


parser = argparse.ArgumentParser(description="Evaluate interactions in NLP models")
parser.add_argument('--gpu_id', default=2, type=int, help="set the gpu id, use -1 to indicate cpu.")
parser.add_argument("--dataset", default="SST-2", type=str, help="set the dataset used.")
parser.add_argument("--arch", default="lstm2_uni", type=str, help="the network architecture.")
parser.add_argument('--data_path', default='/data1/limingjie/data/NLP', type=str, help="root of datasets.")
parser.add_argument("--model_path", default="../saved-models-nlp", type=str, help='where to save the model.')
parser.add_argument("--save_root", default="../saved-interactions", type=str, help='the path of saved interaction.')
parser.add_argument("--baseline_root", default="../saved-baselines", type=str, help="the folder that stores baseline values")

parser.add_argument("--seed", default=0, type=int, help="set the seed used for training model.")
parser.add_argument('--batch_size', default=64, type=int, help="set the batch size for training."
                                                               "This batch size indicates the bs to train the model")
parser.add_argument('--train_lr', default=0.001, type=float, help="set the learning rate for training.")
parser.add_argument("--logspace", default=1, type=int, help='the decay of learning rate.')
parser.add_argument("--epoch", default=200, type=int, help='the number of iterations for training model.')

parser.add_argument("--min_len", type=int, default=6, help="The min length of sentences to eval interaction")
parser.add_argument("--max_len", type=int, default=12, help="The max length of sentences to eval interaction")

parser.add_argument("--selected-dim", default="gt-logistic-odds",
                    help="The value function, can be selected from: gt-log-odds | gt-logistic-odds | gt-prob-log-odds | 0")
parser.add_argument("--baseline-config", default="pad",
                    help="The config of the baseline value")
parser.add_argument("--ci-config", default="precise",
                    help="use the precise/approximated CI (multi-variate interaction)")
parser.add_argument("--eval-sample-num", type=int, default=1,
                    help="evaluate I(S) for ? samples.")

## hparams for customized baseline values (if args.baseline_config == "custom_single")
parser.add_argument("--bound-threshold", type=float, default=0.1, help="bound of baseline value: [_-th*std, _+th*std]")
parser.add_argument("--finetune-lr", type=float, default=0.001)
parser.add_argument("--suppress-ratio", type=float, default=1.0)
parser.add_argument("--finetune-max-iter", type=int, default=50)

args = parser.parse_args()
makedirs_for_eval_model(args)
device = torch.device("cuda") if args.gpu_id >= 0 else torch.device("cpu")

# ===============================================
#   prepare the dataset (for train & eval)
# ===============================================
print()
print('-' * 40)
print("-----------preparing dataset-----------")
print('-' * 40)
train_set, test_set, train_iterator, test_iterator, TEXT, LABEL = load_nlp_dataset(args)


# ===============================================
#   initialize the model
# ===============================================
print()
print('-' * 40)
print("-------------load the model-------------")
print('-' * 40)
net = load_nlp_model(args, TEXT, LABEL, model_path=args.model_path)
net.eval()
# eval_model(args, net, train_iterator, test_iterator)


# ===============================================
#   set the configs, initialize the calculator
# ===============================================
print("------------evaluating CI------------")
if args.baseline_config == "unk":
    raise NotImplementedError
    single_baseline_vector = unk_baseline(net, TEXT, device).to(device)  # [1, emb_dim]
elif args.baseline_config == "pad":
    single_baseline_vector = pad_baseline(net, TEXT, device).to(device)  # [1, emb_dim]
elif args.baseline_config == "custom_single" or args.baseline_config == "custom_single_together":
    single_baseline_vector = None
else:
    raise Exception(f"Unknown baseline config: {args.baseline_config}")

pad = pad_baseline(net, TEXT, device).to(device)

if args.ci_config not in ["precise"]:
    raise Exception(f"Unknown CI config: {args.ci_config}")
interaction_calculator = ConditionedInteraction(mode=args.ci_config)


iterators = {
    "train": train_iterator,
    "test": test_iterator
}

for split in ["train", "test"]:
    cnt = 0
    for idx, batch in enumerate(tqdm(iterators[split], desc=f"Calculating CI ({split} set)")):

        if cnt >= args.eval_sample_num: break
        input, input_length = batch.text
        target = batch.label
        if not (input_length.item() >= args.min_len and input_length.item() <= args.max_len): continue

        print("\n" + "=" * 10 + f"sample-{idx}" + "=" * 10 + "\n")
        split_sample = (split, f"sample-{str(idx).zfill(4)}")
        save_folder = osp.join(args.save_root, *split_sample)
        makedirs(save_folder)

        with torch.no_grad(): input_emb = net.get_emb(input)
        emb_dim = input_emb.shape[2]

        if args.baseline_config in ["unk", "pad"]:
            baseline = torch.cat([single_baseline_vector] * input_length.item(), dim=0).unsqueeze(0)
        elif args.baseline_config == "custom_single":
            baseline_path = osp.join(args.baseline_root, *split_sample, "baseline_final.pth")
            if not osp.exists(baseline_path):
                raise Exception(f"Cannot find the baseline value file at {baseline_path}.")
            baseline = torch.load(osp.join(args.baseline_root, *split_sample, "baseline_final.pth"))
            baseline = baseline.to(device)
        else:
            raise NotImplementedError

        baseline_init = torch.cat([pad] * input_length.item(), dim=0).unsqueeze(0)

        # print("baseline value:", baseline)
        # continue

        # ==========================================
        #       Save basic information
        # ==========================================

        original_sentence = [TEXT.vocab.itos[word] for word in input.squeeze()]
        with open(osp.join(save_folder, "basic-info.txt"), "w") as f:
            f.write("TEXT: " + " ".join(original_sentence) + "\n")
            with torch.no_grad(): output = net(input, input_length)
            f.write(f"output: {output}\n")
            f.write(f"y: {target.item()}\n\n")

            with torch.no_grad():
                output_N = net.emb2out(input_emb, input_length)
                output_empty = net.emb2out(baseline, input_length)
                output_pad = net.emb2out(baseline_init, input_length)
            f.write(f"output_N: {output_N}\n")
            f.write(f"output_empty: {output_empty}\n")
            f.write(f"output_pad: {output_pad}\n\n")

            f.write(f"baseline value: {baseline}\n")
            # f.write(f"input embedding: {input_emb}\n")

        cnt += 1
        # continue

        # ==========================================
        #     Begin calculating the Harsanyi
        # ==========================================

        _, CI_ori = interaction_calculator.calculate(
            net, input_emb, baseline,
            selected_output_dim=args.selected_dim,
            gt=target.item()
        )

        masks, rewards = get_all_subset_rewards(
            net, input_emb, baseline,
            selected_output_dim=args.selected_dim,
            gt=target.item()
        )

        reward2harsanyi = get_reward2harsanyi_mat(dim=input_length).to(device)
        harsanyi2reward = get_harsanyi2reward_mat(dim=input_length).to(device)
        CI = torch.matmul(reward2harsanyi, rewards)

        _, verbose_greedy = remove_noisy_greedy(rewards, CI.clone(), masks, harsanyi2reward, min_patterns=15, n_greedy=40)
        # _, verbose_lsf = remove_noisy_low_strength_first(rewards, CI.clone(), masks, harsanyi2reward, min_patterns=15)
        _, verbose_lsf = remove_noisy_greedy(rewards, CI.clone(), masks, harsanyi2reward, min_patterns=15, n_greedy=1)

        compare_simple_line_chart(data_series=[verbose_greedy["errors"], verbose_lsf["errors"]],
                                  legends=["greedy to min faith(I) [g=40]", "remove lowest-strength I(S)"],
                                  xlabel="# interaction effects removed", ylabel="faith(I)",
                                  title="faith(i)", save_folder=save_folder, save_name="remove_noisy_error")
        compare_simple_line_chart(data_series=[verbose_greedy["explain_ratios"], verbose_lsf["explain_ratios"],],
                                  legends=["greedy to min faith(I) [g=40]", "remove lowest-strength I(S)"],
                                  xlabel="# interaction effects removed", ylabel="explain ratio",
                                  title="explain ratio", save_folder=save_folder, save_name="remove_noisy_explain_ratio")

        CI = CI.cpu().numpy()
        masks = masks.cpu().numpy()

        # ==========================================
        #     Save the calculated results
        # ==========================================

        original_sentence = [TEXT.vocab.itos[word] for word in input.squeeze()]
        with open(osp.join(save_folder, "net-output.txt"), "w") as f:
            f.write("TEXT: " + " ".join(original_sentence) + "\n")
            f.write(f"sum of I(S): {CI.sum()}\n")
            with torch.no_grad(): output = net(input, input_length)
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
        np.save(osp.join(save_folder, "input_emb.npy"), input_emb.squeeze().cpu().numpy())
        np.save(osp.join(save_folder, "baseline.npy"), baseline.squeeze().cpu().numpy())
        save_obj(original_sentence, osp.join(save_folder, "original_sentence_list.bin"))
        save_obj(verbose_greedy["final_retained"], osp.join(save_folder, "final_retained.bin"))
        visualize_all_CI_descending(CI, osp.join(save_folder, "CI_descending.png"))




