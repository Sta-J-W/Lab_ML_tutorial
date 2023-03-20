import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import os.path as osp

from tools.utils import makedirs, load_obj
from set_exp import makedirs_for_visualize_AOG_remove_noisy
from tools.eval_ci import visualize_CI_result, visualize_all_CI_descending, aggregate_pattern_iterative, \
                              visualize_coding_length, plot_input_baseline
from aog import construct_AOG_v1
from tools.metric import eval_explain_ratio_v2_given_coalition_ids




parser = argparse.ArgumentParser(description="Visualize AOG")
parser.add_argument('--gpu_id', default=2, type=int, help="set the gpu id, use -1 to indicate cpu.")
parser.add_argument("--dataset", default="SST-2", type=str, help="set the dataset used.")
parser.add_argument("--arch", default="lstm2_uni", type=str, help="the network architecture.")
parser.add_argument('--data_path', default='/data1/limingjie/data/NLP', type=str, help="root of datasets.")
parser.add_argument('--interaction_root', default='../saved-interactions', help='the pre-calculated interaction files')
parser.add_argument("--save_root", default="../visualized-AOGs", type=str, help='the path of saved AOGs.')

parser.add_argument("--seed", default=0, type=int, help="set the seed used for training model.")
parser.add_argument('--batch_size', default=64, type=int, help="set the batch size for training.")
parser.add_argument('--train_lr', default=0.001, type=float, help="set the learning rate for training.")
parser.add_argument("--logspace", default=1, type=int, help='the decay of learning rate.')
parser.add_argument("--epoch", default=200, type=int, help='the number of iterations for training model.')


parser.add_argument("--selected-dim", default="gt-logistic-odds",
                    help="The value function, can be selected from: gt-log-odds | gt-logistic-odds | gt-prob-log-odds | 0")
parser.add_argument("--baseline-config", default="pad",
                    help="The config of the baseline value")
parser.add_argument("--ci-config", default="precise",
                    help="use the precise/approximated CI (multi-variate interaction)")


## hparams for customized baseline values (if args.baseline_config == "custom_single")
parser.add_argument("--bound-threshold", type=float, default=0.1, help="bound of baseline value: [_-th*std, _+th*std]")
parser.add_argument("--finetune-lr", type=float, default=0.001)
parser.add_argument("--suppress-ratio", type=float, default=1.0)
parser.add_argument("--finetune-max-iter", type=int, default=50)

# hparams for AOGs
parser.add_argument('--objective', default='10.0_entropy+total_length-eff-early')
parser.add_argument('--max-n-merges', type=int, default=5)


args = parser.parse_args()
makedirs_for_visualize_AOG_remove_noisy(args)
device = torch.device("cuda") if args.gpu_id >= 0 else torch.device("cpu")


if args.dataset == "SST-2":
    label_name = {0: "positive sentiment", 1: "negative sentiment"}
    get_label_name = lambda label: label_name[label]
    get_pred_label = lambda output, label: label if output > 0 else int(not bool(label))
elif args.dataset == "CoLA":
    label_name = {0: "grammatically correct", 1: "grammatically wrong"}
    get_label_name = lambda label: label_name[label]
    get_pred_label = lambda output, label: label if output > 0 else int(not bool(label))
else:
    raise NotImplementedError


# ===============================================
#   If there are mannual configs
# ===============================================
selected_samples = None
fig_sizes = None
max_node_each_row = None
simplify_fig_sizes = None


def judge_is_selected(attribute, attribute_id, concept):
    return concept[attribute_id]

splits = ["train", "test"]
# splits = ["test"]
for split in splits:
    # the folder to load the pre-calculated CI
    result_folder = osp.join(args.interaction_root, split)

    samples = sorted(os.listdir(result_folder))
    for sample in samples:
        sample_id = int(sample.split("-")[-1])
        if selected_samples is not None and sample_id not in selected_samples[split]:
            continue

        sample_save_folder = osp.join(args.save_root, split, sample)
        makedirs(sample_save_folder)

        print(f" === plotting sample {sample} in {split} split. ===")
        words = load_obj(osp.join(result_folder, sample, "original_sentence_list.bin"))
        masks = np.load(osp.join(result_folder, sample, "masks.npy"))
        retained = load_obj(osp.join(result_folder, sample, "final_retained.bin"))
        not_empty = np.any(masks, axis=1)
        CI = np.load(osp.join(result_folder, sample, "CI.npy"))
        v_S = CI.sum()
        with open(osp.join(result_folder, sample, "net-output.txt"), 'r') as f:
            info = f.read().strip().split("\n")
            label_line = None
            for line in info:
                if line.startswith("y:"):
                    label_line = line
                    break
            label = int(eval(label_line.strip().split(":")[-1]))

        retained = sorted(retained, key=lambda idx: -abs(CI[idx]))

        with open(osp.join(sample_save_folder, "interactions.txt"), "w") as f:
            for i in range(len(retained)):
                f.write(str(masks[retained[i]]) + "\t")
                f.write(str(CI[retained[i]]) + "\n")

        attributes_baselines = words

        visualize_all_CI_descending(CI, osp.join(sample_save_folder, f"{sample}-all.png"))

        # save the info of the AOG
        explain_ratio_v2 = eval_explain_ratio_v2_given_coalition_ids(CI, masks, retained)
        with open(osp.join(sample_save_folder, "aog_info.txt"), "w") as f:
            f.write(f"sentence: {' '.join(words)}\n")
            f.write(f"# patterns:\t{len(retained)}\n")
            f.write(f"explain ratio v2: {explain_ratio_v2}\n")
            f.write(f"v(S)={CI.sum()} | g(S)={CI[retained].sum()}\n\n")
            f.write('\n'.join(info))

        # CI_order = CI_order[:plot_num]
        visualize_CI_result(
            attributes=words, concepts=masks[retained],
            eval_val=CI[retained], eval_std=np.zeros_like(CI[retained]),
            eval_type=r"$\mathbb{E}_{data}[I(S)]$" + f" v(S)={v_S:.2f}",
            judge_is_selected=judge_is_selected,
            save_path=osp.join(sample_save_folder, f"{sample}-top-all.png")
        )

        max_iter = args.max_n_merges

        # merge the frequently-occurred sub-patterns
        merged_patterns, aggregated_concepts, coding_length = aggregate_pattern_iterative(
            concepts=masks[retained], eval_val=CI[retained], max_iter=max_iter, objective=args.objective
        )
        for key in coding_length.keys():
            visualize_coding_length(coding_length[key],
                                    osp.join(sample_save_folder, f"{sample}-coding-length-{key}-top-{len(retained)}.png"))

        aggregated_attributes = []
        for pattern in merged_patterns:
            description = []
            for i in range(pattern.shape[0]):
                if pattern[i]: description.append(words[i])
            description = "+".join(description)
            aggregated_attributes.append(description)

        visualize_CI_result(
            attributes=words + aggregated_attributes, concepts=aggregated_concepts,
            eval_val=CI[retained], eval_std=np.zeros_like(CI[retained]),
            eval_type=r"$\mathbb{E}_{data}[CI(S)]$ in " + sample,
            judge_is_selected=judge_is_selected,
            save_path=osp.join(sample_save_folder, f"{sample}-aggregated-{max_iter}.png"), figsize=(20, 10)
        )

        single_features = np.vstack([np.eye(len(words)).astype(bool), merged_patterns])

        aog = construct_AOG_v1(
            attributes=attributes_baselines,
            attributes_baselines=attributes_baselines,
            single_features=single_features,
            concepts=aggregated_concepts,
            eval_val=CI[retained]
        )

        figsize = (20, 8) if fig_sizes is None else fig_sizes[split][sample_id]
        sfigsize = (7, 3.5) if simplify_fig_sizes is None else simplify_fig_sizes[split][sample_id]
        max_node = 10 if max_node_each_row is None else max_node_each_row[split][sample_id]

        aog.visualize(
            save_path=osp.join(sample_save_folder, f"{sample}-aog-{max_iter}.html"), figsize=figsize,
            renderer="networkx", n_row_interaction=int(np.ceil(len(retained) / max_node)),
            title=f"output={CI[retained].sum():.2f} "
                  f"| prediction: {get_label_name(get_pred_label(output=CI[retained].sum(), label=label))} "
                  f"| R={100 * explain_ratio_v2:.2f}%"
        )

        aog.visualize(
            save_path=osp.join(sample_save_folder, f"{sample}-aog-{max_iter}-highlight-1.svg"), figsize=figsize,
            renderer="networkx", n_row_interaction=int(np.ceil(len(retained) / max_node)), highlight_path=f"rank-1",
        )
