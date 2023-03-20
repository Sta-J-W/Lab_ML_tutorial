import torch
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import argparse
from baseline.learn_baseline import learn_baseline_suppress_IS_single
from tools.utils import get_all_emb, makedirs, save_obj
from set_exp import makedirs_for_finetune_baseline_single
from model import load_nlp_model
from dataset import load_nlp_dataset
from baseline import unk_baseline, pad_baseline
from interaction import ConditionedInteraction

parser = argparse.ArgumentParser(description="Finetune baseline values for NLP models")
parser.add_argument('--gpu_id', default=1, type=int, help="set the gpu id, use -1 to indicate cpu.")
parser.add_argument("--dataset", default="SST-2", type=str, help="set the dataset used.")
parser.add_argument("--arch", default="cnn", type=str, help="the network architecture.")
parser.add_argument('--data_path', default='/data1/limingjie/data/NLP', type=str, help="root of datasets.")
parser.add_argument("--model_path", default="../saved-models-nlp", type=str, help='where to save the model.')
parser.add_argument("--save_root", default="../saved-baselines", type=str, help='the path of baselines.')

parser.add_argument("--seed", default=0, type=int, help="set the seed used for training model.")
parser.add_argument('--batch_size', default=64, type=int, help="set the batch size for training.")
parser.add_argument('--train_lr', default=0.001, type=float, help="set the learning rate for training.")
parser.add_argument("--logspace", default=1, type=int, help='the decay of learning rate.')
parser.add_argument("--epoch", default=200, type=int, help='the number of iterations for training model.')

parser.add_argument("--min_len", type=int, default=6, help="The min length of sentences to eval interaction")
parser.add_argument("--max_len", type=int, default=12, help="The max length of sentences to eval interaction")

parser.add_argument("--selected-dim", default="gt-logistic-odds",
                    help="The value function, can be selected from: gt-log-odds | gt-logistic-odds | gt-prob-log-odds | 0")
parser.add_argument("--baseline-config", default="pad", help="The initialization of the baseline value")
parser.add_argument("--ci-config", default="precise", help="use the precise/approximated CI (multi-variate interaction)")

## the settings for finetuning the baselines
parser.add_argument("--bound-threshold", type=float, default=0.1, help="bound of baseline value: [_-th*std, _+th*std]")
parser.add_argument("--finetune-lr", type=float, default=0.001)
parser.add_argument("--suppress-ratio", type=float, default=1.0)
parser.add_argument("--finetune-max-iter", type=int, default=50)
parser.add_argument("--finetune-val-freq", type=int, default=1,
                    help="how many epochs to validate the distribution of I(S)")
parser.add_argument("--eval-sample-num", type=int, default=1,
                    help="evaluate I(S) for ? samples.")

args = parser.parse_args()
makedirs_for_finetune_baseline_single(args)
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
#   initialize the baseline value
# ===============================================
if args.baseline_config == "unk":
    raise Exception
    single_baseline_vector = unk_baseline(net, TEXT, device).to(device)  # [1, emb_dim]
elif args.baseline_config == "pad":
    single_baseline_vector = pad_baseline(net, TEXT, device).to(device)  # [1, emb_dim]
else:
    raise Exception(f"Unknown baseline config: {args.baseline_config}")

# ===============================================
#   initialize the calculator
# ===============================================
if args.ci_config not in ["precise"]:
    raise Exception(f"Unknown CI config: {args.ci_config}")
interaction_calculator = ConditionedInteraction(mode=args.ci_config)

# ===============================================
#   data for calculating I(S)
# ===============================================
iterators = {
    "train": train_iterator,
    "test": test_iterator
}

# ====================================================
#     the min and max of baselines
# ====================================================
embs = get_all_emb(net, TEXT, device)  # [n_vocab, emb_dim]
bound_threshold = args.bound_threshold
min_baseline_vector = single_baseline_vector - bound_threshold * torch.std(embs, dim=0, keepdim=True)  # [1, emb_dim]
max_baseline_vector = single_baseline_vector + bound_threshold * torch.std(embs, dim=0, keepdim=True)  # [1, emb_dim]

for split in ["train", "test"]:
    cnt = 0
    for idx, batch in enumerate(tqdm(iterators[split], desc=f"Calculating CI ({split} set)")):
        if cnt >= args.eval_sample_num: break

        input, input_length = batch.text
        target = batch.label
        if not (input_length.item() >= args.min_len and input_length.item() <= args.max_len): continue

        split_sample = (split, f"sample-{str(idx).zfill(4)}")

        with torch.no_grad(): input_emb = net.get_emb(input)
        emb_dim = input_emb.shape[2]

        baseline_init = torch.cat([single_baseline_vector] * input_length.item(), dim=0).unsqueeze(0)
        baseline_max = torch.cat([max_baseline_vector] * input_length.item(), dim=0).unsqueeze(0)
        baseline_min = torch.cat([min_baseline_vector] * input_length.item(), dim=0).unsqueeze(0)

        masks, CI = interaction_calculator.calculate(
            net, input_emb, baseline_init,
            selected_output_dim=args.selected_dim,
            gt=target.item()
        )

        CI = CI.cpu().numpy()
        masks = masks.cpu().numpy()
        save_folder = osp.join(args.save_root, *split_sample)
        makedirs(save_folder)
        print("save at:", save_folder)
        np.save(osp.join(save_folder, "CI-init.npy"), CI)

        original_sentence = [TEXT.vocab.itos[word] for word in input.squeeze()]
        with open(osp.join(save_folder, "net-output.txt"), "w") as f:
            f.write("TEXT: " + " ".join(original_sentence) + "\n")
            f.write(f"sum of I(S): {CI.sum()}\n")
            with torch.no_grad(): output = net(input, input_length)
            f.write(f"v(N): {output}\n")
            f.write(f"y: {target.item()}\n")

        learn_baseline_suppress_IS_single(
            model=net, embedded=input_emb, y=target, suppress_ratio=args.suppress_ratio,
            baseline_init=baseline_init, baseline_max=baseline_max, baseline_min=baseline_min,
            CI_sample=CI, all_masks=masks, ci_config=args.ci_config, selected_dim=args.selected_dim,
            baseline_lr=args.finetune_lr, max_iter=args.finetune_max_iter, val_freq=args.finetune_val_freq,
            device=device, save_root=save_folder
        )
        cnt += 1

