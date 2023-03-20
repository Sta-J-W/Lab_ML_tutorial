import json
import os
import os.path as osp
import shutil

import numpy as np
import argparse

import torch

from harsanyi.and_or_harsanyi_utils import get_reward2Iand_mat, get_Iand2reward_mat
from harsanyi.interaction_utils import flatten

from tools.plot import visualize_all_interaction_descending, plot_coalition
from tools.metric import eval_explain_ratio_v2_given_coalition_ids
from tools.remove_noisy import remove_noisy_greedy
from tools.eval_ci import visualize_CI_result, aggregate_pattern_iterative, visualize_coding_length
from aog import construct_AOG_v1


parser = argparse.ArgumentParser("Visualize the And-Or Graph (AOG)")
parser.add_argument("--result-root", type=str, default="../saved-results/finetune-baseline",
                    help="the root folder that stores the finetune result")
parser.add_argument("--segment-root", default="../saved-manual-segments", type=str,
                    help="the root folder for storing the segmentation info")
parser.add_argument("--model-args", type=str, default=None,
                    help="the args for the trained model")
parser.add_argument("--segment-args", type=str, default=None,
                    help="the args for the manual segments")
parser.add_argument("--harsanyi-args", type=str, default=None,
                    help="the args for the computed harsanyi dividends")
parser.add_argument('--objective', default='10.0_entropy+total_length-eff-early')
parser.add_argument("--gpu-id", type=int, default=3)


args = parser.parse_args()


def judge_is_selected(attribute, attribute_id, concept):
    return concept[attribute_id]


result_root: str = args.result_root
segment_root: str = args.segment_root
model_args: str = args.model_args
segment_args: str = args.segment_args
harsanyi_args: str = args.harsanyi_args
assert model_args is not None and segment_args is not None and harsanyi_args is not None

result_folder = osp.join(result_root, model_args, segment_args, harsanyi_args)

for class_id in sorted(os.listdir(result_folder)):
    if not osp.isdir(osp.join(result_folder, class_id)): continue
    for sample_id in sorted(os.listdir(osp.join(result_folder, class_id))):
        if sample_id != "sample_00002": continue
        sample_save_folder = osp.join(result_folder, class_id, sample_id, "aog")
        if osp.exists(sample_save_folder):
            shutil.rmtree(sample_save_folder)
        os.makedirs(sample_save_folder, exist_ok=True)

        print("Plotting", sample_id)

        # ===========================================
        #
        # ===========================================
        rewards = np.load(osp.join(result_folder, class_id, sample_id, "after_sparsify", "rewards.npy"))
        I_and = np.load(osp.join(result_folder, class_id, sample_id, "after_sparsify", "Iand.npy"))
        masks = np.load(osp.join(result_folder, class_id, sample_id, "after_sparsify", "masks.npy"))
        image = torch.load(osp.join(result_folder, class_id, sample_id, "image.pth"))
        with open(osp.join(result_folder, class_id, sample_id, "all_players.json"), "r") as f:
            all_players = json.load(f)
        n_dim = masks.shape[1]
        visualize_all_interaction_descending(I_and, osp.join(sample_save_folder, f"all-interactions.png"))

        rewards = torch.from_numpy(rewards).float().to(args.gpu_id)
        I_and = torch.from_numpy(I_and).float().to(args.gpu_id)
        masks = torch.from_numpy(masks).bool().to(args.gpu_id)

        reward2Iand = get_reward2Iand_mat(n_dim).float().to(args.gpu_id)
        Iand2reward = get_Iand2reward_mat(n_dim).float().to(args.gpu_id)

        # ===========================================
        #
        # ===========================================
        _, verbose_greedy = remove_noisy_greedy(rewards, I_and.clone(), masks, Iand2reward, min_patterns=20, n_greedy=40,
                                                thres_approx_error=np.sqrt(0.05), thres_explain_ratio=0.98)
        retained = verbose_greedy["final_retained"]
        retained = sorted(retained, key=lambda idx: -abs(I_and[idx]))
        if len(retained) > 40: continue
        attributes = [r"$x_{" + str(i) + r"}$" for i in range(1, 1+n_dim)]

        # save the info of the AOG
        explain_ratio_v2 = eval_explain_ratio_v2_given_coalition_ids(I_and, masks, retained)
        with open(osp.join(sample_save_folder, "aog_info.txt"), "w") as f:
            f.write(f"# patterns:\t{len(retained)}\n")
            f.write(f"explain ratio v2: {explain_ratio_v2}\n")
            f.write(f"v(N)={I_and.sum()} | g(N)={I_and[retained].sum()}\n\n")


        # ==================================================
        #     generate and-or graph structure
        # ==================================================

        I_and = I_and.cpu().numpy()
        rewards = rewards.cpu().numpy()
        masks = masks.cpu().numpy()

        visualize_CI_result(
            attributes=attributes, concepts=masks[retained],
            eval_val=I_and[retained], eval_std=np.zeros_like(I_and[retained]),
            eval_type=r"$I(S)$ " + f" v(N)={I_and.sum():.2f}",
            judge_is_selected=judge_is_selected,
            save_path=osp.join(sample_save_folder, f"top-interactions.png")
        )

        max_iter = 5

        # merge the frequently-occurred sub-patterns
        merged_patterns, aggregated_concepts, coding_length = aggregate_pattern_iterative(
            concepts=masks[retained], eval_val=I_and[retained], max_iter=max_iter, objective=args.objective
        )
        for key in coding_length.keys():
            visualize_coding_length(
                coding_length[key],
                osp.join(sample_save_folder, f"coding-length-{key}-top-{len(retained)}.png")
            )

        aggregated_attributes = []
        for pattern in merged_patterns:
            description = []
            for i in range(pattern.shape[0]):
                if pattern[i]: description.append(attributes[i])
            description = "+".join(description)
            aggregated_attributes.append(description)

        visualize_CI_result(
            attributes=attributes + aggregated_attributes, concepts=aggregated_concepts,
            eval_val=I_and[retained], eval_std=np.zeros_like(I_and[retained]),
            eval_type=r"$I(S)$ of " + sample_id,
            judge_is_selected=judge_is_selected,
            save_path=osp.join(sample_save_folder, f"aggregated-{max_iter}.png"), figsize=(20, 10)
        )

        single_features = np.vstack([np.eye(len(attributes)).astype(bool), merged_patterns])

        aog = construct_AOG_v1(
            attributes=attributes,
            attributes_baselines=attributes,
            single_features=single_features,
            concepts=aggregated_concepts,
            eval_val=I_and[retained],
        )


        max_node = 8
        n_row_interaction = int(np.ceil(len(retained) / max_node))
        if n_row_interaction == 3:
            figsize = (16, 8.5)
        elif n_row_interaction == 4:
            figsize = (16, 10)
        elif n_row_interaction == 5:
            figsize = (16, 11)
        else:
            figsize = (16, 15)

        aog.visualize(
            save_path=osp.join(sample_save_folder, f"aog.html"), figsize=figsize,
            renderer="networkx", n_row_interaction=n_row_interaction,
            title=f"output={I_and[retained].sum():.2f} "
                  f"| R={100 * explain_ratio_v2:.2f}%"
        )

        for i in range(1, 11):
            aog.visualize(
                save_path=osp.join(sample_save_folder, f"aog-{max_iter}-highlight-{i}.svg"),
                renderer="networkx", n_row_interaction=n_row_interaction,
                highlight_path=f"rank-{i}", figsize=figsize
            )

        plot_image = image.squeeze(0).clone().cpu().numpy()
        # visualize players
        for player_id, player in enumerate(all_players):
            plot_coalition(
                image=plot_image, grid_width=1, coalition=[flatten([player])],
                save_folder=osp.join(sample_save_folder, "players"), save_name=f"player_{player_id}",
                fontsize=15, linewidth=4, figsize=(4, 4), alpha=0.6, save_format='svg'
            )
        # visualize coalitions
        all_players = np.array(all_players, dtype=object)
        for pattern_id, pattern in enumerate(merged_patterns):
            plot_coalition(
                image=plot_image, grid_width=1, coalition=[flatten(all_players[pattern])],
                save_folder=osp.join(sample_save_folder, "players"), save_name=f"pattern_{pattern_id}",
                fontsize=15, linewidth=4, figsize=(4, 4), alpha=0.6, save_format='svg'
            )
        # visualize top-10 concepts
        for concept_id, i in enumerate(retained[:11]):
            plot_coalition(
                image=plot_image, grid_width=1, coalition=[flatten(all_players[masks[i]])],
                save_folder=osp.join(sample_save_folder, "players"), save_name=f"concept_{concept_id}",
                fontsize=15, linewidth=4, figsize=(4, 4), alpha=0.6, save_format='svg'
            )

