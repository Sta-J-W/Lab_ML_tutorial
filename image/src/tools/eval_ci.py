import os
import os.path as osp
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from harsanyi.interaction_utils import generate_all_masks, is_A_subset_B, select_subset, set_to_index, get_subset, is_A_subset_Bs
from pprint import pprint

def get_CI_mean_std(result_folder, split=None, category=None):
    if split is None:
        splits = ["train", "test"]
    else:
        splits = [split]

    CI_all_sample = []
    masks = None
    for split in splits:
        result_folder = osp.join(result_folder, split)
        if category is None:
            categories = os.listdir(result_folder)
        else:
            categories = [category]
        for category in categories:
            samples = os.listdir(osp.join(result_folder, category))
            for sample in samples:
                masks = np.load(osp.join(result_folder, category, sample, "masks.npy"))
                CI = np.load(osp.join(result_folder, category, sample, "CI.npy"))
                CI_all_sample.append(CI.copy())

    CI_all_sample = np.stack(CI_all_sample)
    CI_mean = np.mean(CI_all_sample, axis=0)
    CI_std = np.std(CI_all_sample, axis=0)
    return CI_mean, CI_std, masks


def visualize_CI_result(attributes, concepts, eval_val, eval_std, eval_type, judge_is_selected, save_path="test.png", **kwargs):
    if "figsize" in kwargs.keys():
        plt.figure(figsize=kwargs["figsize"])
    else:
        plt.figure(figsize=(15, 6))

    plt.subplot(2, 1, 2)
    ax_attribute = plt.gca()
    x = np.arange(len(concepts))
    y = np.arange(len(attributes))
    # plt.xticks(x, concepts)
    plt.xticks(x, [])
    plt.yticks(y, attributes)
    plt.xlim(x.min() - 0.5, x.max() + 0.5)
    plt.ylim(y.min() - 0.5, y.max() + 0.5)
    plt.xlabel("concept")
    plt.ylabel("attribute")

    patch_colors = {
        True: {
            'pos': 'red',
            'neg': 'blue'
        },
        False: 'gray'
    }
    patch_width = 0.8
    patch_height = 0.9

    for concept_id in range(len(concepts)):
        concept = concepts[concept_id]
        for attribute_id in range(len(attributes)):
            is_selected = judge_is_selected(attributes[attribute_id], attribute_id, concept)
            if not is_selected:
                facecolor = patch_colors[is_selected]
            else:
                if eval_val[concept_id] > 0: facecolor = patch_colors[is_selected]['pos']
                else: facecolor = patch_colors[is_selected]['neg']
            rect = Rectangle(
                xy=(concept_id - patch_width / 2,
                    attribute_id - patch_height / 2),
                width=patch_width, height=patch_height,
                edgecolor=None,
                facecolor=facecolor,
                alpha=0.5
            )
            ax_attribute.add_patch(rect)

    plt.subplot(2, 1, 1, sharex=ax_attribute)
    plt.ylabel(eval_type)
    # plt.yscale("log")
    ax_eval = plt.gca()
    plt.setp(ax_eval.get_xticklabels(), visible=False)
    ax_eval.spines['right'].set_visible(False)
    ax_eval.spines['top'].set_visible(False)
    # plt.errorbar(np.arange(len(concepts)), eval_val, yerr=eval_std)
    plt.errorbar(np.arange(len(concepts)), np.abs(eval_val), yerr=eval_std)
    plt.hlines(y=0, xmin=0, xmax=len(concepts), linestyles='dotted', colors='red')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")


def visualize_all_CI_descending(all_ci, save_path="test.png"):
    plt.figure(figsize=(8, 6))
    X = np.arange(1, all_ci.shape[0] + 1)
    plt.hlines(0, 0, X.shape[0], linestyles="dotted", colors="red")
    plt.plot(X, all_ci[np.argsort(-all_ci)])
    plt.xlabel("patterns (with I(S) descending)")
    plt.ylabel("I(S)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close("all")


def visualize_CI_IOU(attributes, concepts, eval_val, eval_val_others, eval_type, judge_is_selected, save_path="test.png"):
    plt.figure(figsize=(15, 6))

    plt.subplot(2, 1, 2)
    ax_attribute = plt.gca()
    x = np.arange(len(concepts))
    y = np.arange(len(attributes))
    # plt.xticks(x, concepts)
    plt.xticks(x, [])
    plt.yticks(y, attributes)
    plt.xlim(x.min() - 0.5, x.max() + 0.5)
    plt.ylim(y.min() - 0.5, y.max() + 0.5)
    plt.xlabel("concept")
    plt.ylabel("attribute")

    patch_colors = {
        True: {
            'pos': 'red',
            'neg': 'blue'
        },
        False: 'gray'
    }
    patch_width = 0.8
    patch_height = 0.9

    for concept_id in range(len(concepts)):
        concept = concepts[concept_id]
        for attribute_id in range(len(attributes)):
            is_selected = judge_is_selected(attributes[attribute_id], attribute_id, concept)
            if not is_selected:
                facecolor = patch_colors[is_selected]
            else:
                if eval_val[concept_id] > 0:
                    facecolor = patch_colors[is_selected]['pos']
                else:
                    facecolor = patch_colors[is_selected]['neg']
            rect = Rectangle(
                xy=(concept_id - patch_width / 2,
                    attribute_id - patch_height / 2),
                width=patch_width, height=patch_height,
                edgecolor=None,
                facecolor=facecolor,
                alpha=0.5
            )
            ax_attribute.add_patch(rect)

    plt.subplot(2, 1, 1, sharex=ax_attribute)
    plt.ylabel(eval_type)
    # plt.yscale("log")
    ax_eval = plt.gca()
    plt.setp(ax_eval.get_xticklabels(), visible=False)
    ax_eval.spines['right'].set_visible(False)
    ax_eval.spines['top'].set_visible(False)
    plt.plot(np.arange(len(concepts)), eval_val, label="model 0")
    for i, eval_val_other in enumerate(eval_val_others):
        plt.plot(np.arange(len(concepts)), eval_val_other, label=f"model {i+1}")
    # plt.plot(np.arange(len(concepts)), np.minimum(eval_val, eval_val_other), label="min")
    plt.hlines(y=0, xmin=0, xmax=len(concepts), linestyles='dotted', colors='red')
    plt.legend(ncol=5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")


def visualize_CI_IOU_strength(attributes, concepts, eval_val, eval_val_others, eval_type, judge_is_selected, save_path="test.png", plot_min=False, title=None):
    plt.figure(figsize=(15, 6))

    plt.subplot(2, 1, 2)
    ax_attribute = plt.gca()
    x = np.arange(len(concepts))
    y = np.arange(len(attributes))
    # plt.xticks(x, concepts)
    plt.xticks(x, [])
    plt.yticks(y, attributes)
    plt.xlim(x.min() - 0.5, x.max() + 0.5)
    plt.ylim(y.min() - 0.5, y.max() + 0.5)
    plt.xlabel("concept")
    plt.ylabel("attribute")

    patch_colors = {
        True: {
            'pos': 'red',
            'neg': 'blue'
        },
        False: 'gray'
    }
    patch_width = 0.8
    patch_height = 0.9

    for concept_id in range(len(concepts)):
        concept = concepts[concept_id]
        for attribute_id in range(len(attributes)):
            is_selected = judge_is_selected(attributes[attribute_id], attribute_id, concept)
            if not is_selected:
                facecolor = patch_colors[is_selected]
            else:
                if eval_val[concept_id] > 0:
                    facecolor = patch_colors[is_selected]['pos']
                else:
                    facecolor = patch_colors[is_selected]['neg']
            rect = Rectangle(
                xy=(concept_id - patch_width / 2,
                    attribute_id - patch_height / 2),
                width=patch_width, height=patch_height,
                edgecolor=None,
                facecolor=facecolor,
                alpha=0.5
            )
            ax_attribute.add_patch(rect)

    plt.subplot(2, 1, 1, sharex=ax_attribute)
    plt.ylabel(eval_type)
    # plt.yscale("log")
    ax_eval = plt.gca()
    plt.setp(ax_eval.get_xticklabels(), visible=False)
    ax_eval.spines['right'].set_visible(False)
    ax_eval.spines['top'].set_visible(False)
    plt.plot(np.arange(len(concepts)), eval_val, linewidth=4.0, label="average strength")
    for i, eval_val_other in enumerate(eval_val_others):
        plt.plot(np.arange(len(concepts)), eval_val_other,  label=f"model {i}")
    if plot_min:
        plt.plot(np.arange(len(concepts)), np.min(eval_val_others, 0), linewidth=4.0, label="min")
        IoU = np.min(eval_val_others,0).sum()/np.max(eval_val_others,0).sum()
        print("IoU: {}".format(IoU))

    if title is not None:
        plt.title(title, fontdict={"fontsize": 15, "weight": "bold"})
    plt.hlines(y=0, xmin=0, xmax=len(concepts), linestyles='dotted', colors='red')
    plt.legend(ncol=5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")


# def aggregate_pattern(concepts, eval_val, max_num=20):
#     n_feature_dim = concepts.shape[1]
#     all_concepts = np.array(generate_all_masks(n_feature_dim))
#     frequency = {set_to_index(concept): [concept, 0] for concept in all_concepts}
#     assert len(frequency) == all_concepts.shape[0]
#     for concept, CI in zip(concepts, eval_val):
#         sub_concepts = select_subset(all_concepts, concept)
#         for sub_concept in sub_concepts:
#             sub_concept_idx = set_to_index(sub_concept)
#             if sub_concept_idx == 0: continue  # empty set
#             frequency[sub_concept_idx][1] += CI
#
#     pattern_frequency = sorted(list(frequency.values()), key=lambda concept_freq: concept_freq[1], reverse=True)[:max_num]
#     return pattern_frequency



def calculate_total_code_length(concepts, eval_val):
    eps = 1e-8
    frequency = np.matmul(eval_val, concepts)
    frequency /= (frequency.sum() + eps)
    codeword_length = np.log(1 / (frequency + eps))
    total_code_length = np.dot(np.matmul(eval_val, concepts), codeword_length)
    return total_code_length


def calculate_average_code_length(concepts, eval_val):
    eps = 1e-8
    frequency = np.matmul(eval_val, concepts)
    frequency /= (frequency.sum() + eps)
    codeword_length = np.log(1 / (frequency + eps))
    average_code_length = np.dot(frequency, codeword_length)
    return average_code_length



def calculate_codebook_length(
        original_concepts: np.ndarray,  # [n_patterns, n_attributes]
        merged_concepts: np.ndarray,  # [n_patterns, n_merged]
        codebook: np.ndarray,  # [n_merged, n_attributes]
        eval_val: np.ndarray  # [n_patterns,]
):
    eps = 1e-8
    is_codeword_used = np.any(merged_concepts, axis=0)  # whether the merged patterns are used [n_merged, n_attributes]
    # 1. the coding length of the itemset
    item_frequency = np.matmul(eval_val, original_concepts)
    item_frequency /= (item_frequency.sum() + eps)
    item_code_length = np.log(1 / (item_frequency + eps))  # [n_attributes,]
    itemset_code_length = np.matmul(codebook, item_code_length)
    itemset_code_length = itemset_code_length[is_codeword_used].sum()
    # 2. the coding length of codewords
    codeword_frequency = np.matmul(eval_val, merged_concepts)
    codeword_frequency /= (codeword_frequency.sum() + eps)
    codewords_code_length = np.log(1 / (codeword_frequency + eps))[is_codeword_used].sum()
    # 3. the coding length of the codebook is the sum of (1) and (2)
    codebook_complexity = itemset_code_length + codewords_code_length
    return codebook_complexity



def aggregate_pattern_iterative(
        concepts: np.ndarray,
        eval_val: np.ndarray,
        max_iter: int = 20,
        objective: str = "total_length",
        early_stop: bool = True
):
    '''
    This function will aggregate feature dimensions to form new 'code words', to shorten the code length.
    :param concepts: <numpy.ndarray> (n_concepts, n_features) -- an array of concepts (masks)
    :param eval_val: <numpy.ndarray> (n_concepts, ) -- the interaction of these concepts
    :param max_iter: the maximum number of merges
    :param objective: str -- "total_length" or "avg_length"
    :return: the merged codewords, the new concept mask, code length during optimization
    '''
    eval_val = np.abs(eval_val.copy())
    original_concepts = concepts.copy()
    n_feature_dim = concepts.shape[1]
    codebook = np.eye(n_feature_dim).astype(bool)

    if objective == "total_length":
        # calculate_code_length = calculate_total_code_length
        calculate_code_length = \
            lambda ori_concepts, mer_concepts, cdbk, val: calculate_total_code_length(mer_concepts, val)
        raise Exception(f"The objective {objective} is now deprecated.")
    elif objective == "avg_length":
        calculate_code_length = calculate_average_code_length
        raise Exception(f"The objective {objective} is now deprecated.")
    elif objective == "codebook+total_length":
        calculate_code_length = \
            lambda ori_concepts, mer_concepts, cdbk, val: calculate_total_code_length(mer_concepts, val) \
                                      + calculate_codebook_length(ori_concepts, mer_concepts, cdbk, val)
    elif objective.endswith("entropy+total_length"):  # "xx.xx_entropy+total_length"
        entropy_lamb = float(objective.split("_")[0])
        calculate_code_length = \
            lambda ori_concepts, mer_concepts, cdbk, val: calculate_total_code_length(mer_concepts, val) \
                                       + entropy_lamb * calculate_average_code_length(mer_concepts, val)
    elif objective.endswith("entropy+total_length-eff") or objective.endswith("entropy+total_length-eff-early"):  # for effective descent
        entropy_lamb = float(objective.split("_")[0])
        calculate_code_length = \
            lambda ori_concepts, mer_concepts, cdbk, val: calculate_total_code_length(mer_concepts, val) \
                                       + entropy_lamb * calculate_average_code_length(mer_concepts, val)
    else:
        raise Exception(f"Unknown objective: {objective}")

    code_lengths = {
        "sum": [],
    }

    code_length = calculate_code_length(original_concepts, concepts, codebook, eval_val)
    code_lengths["sum"].append(code_length)

    if objective == "codebook+total_length":
        code_lengths["words"] = [calculate_total_code_length(concepts, eval_val)]
        code_lengths["codebook"] = [calculate_codebook_length(original_concepts, concepts, codebook, eval_val)]
    elif objective.endswith("entropy+total_length"):
        code_lengths["words"] = [calculate_total_code_length(concepts, eval_val)]
        code_lengths["entropy"] = [calculate_average_code_length(concepts, eval_val)]
    elif objective.endswith("entropy+total_length-eff") or objective.endswith("entropy+total_length-eff-early"):
        code_lengths["words"] = [calculate_total_code_length(concepts, eval_val)]
        code_lengths["entropy"] = [calculate_average_code_length(concepts, eval_val)]
        code_lengths["decrease-per-dim"] = []
        code_lengths["merge-size"] = []

    for _ in range(max_iter):
        length_after_merge = {}

        for concept in concepts:
            sub_concepts = get_subset(concept)
            for sub_concept in sub_concepts:
                if sub_concept.sum() < 2: continue  # if aiming to merge 0 or 1 code words ...
                sub_concept_idx = set_to_index(sub_concept)
                if sub_concept_idx in length_after_merge.keys(): continue
                # judge if each concept in 'concepts' has such combination
                flag = is_A_subset_Bs(sub_concept, concepts)

                concepts_after_merge = concepts.copy()
                indice = np.outer(flag, sub_concept).astype(bool)
                concepts_after_merge[indice] = False
                concepts_after_merge = np.hstack([concepts_after_merge, flag.reshape(-1, 1)])
                codebook_after_merge = np.vstack([codebook, np.any(codebook[sub_concept], axis=0)])

                code_length = calculate_code_length(original_concepts, concepts_after_merge, codebook_after_merge, eval_val)
                length_after_merge[sub_concept_idx] = [code_length, sub_concept, concepts_after_merge, codebook_after_merge]

        if len(length_after_merge.keys()) == 0: break

        # select the merge
        if objective.endswith("entropy+total_length-eff"):
            prev_code_length = code_lengths["sum"][-1]
            code_length, _, _, _ = sorted(
                list(length_after_merge.values()), key=lambda item: (item[0] - prev_code_length) / np.sum(item[3][-1]),
            )[0]
            if early_stop and code_length > code_lengths["sum"][-1]: break
            code_length, _, concepts, codebook = sorted(
                list(length_after_merge.values()), key=lambda item: (item[0] - prev_code_length) / np.sum(item[3][-1]),
            )[0]
            code_lengths["sum"].append(code_length)
        elif objective.endswith("entropy+total_length-eff-early"):
            prev_code_length = code_lengths["sum"][-1]
            code_length, _, concepts_new, _ = sorted(
                list(length_after_merge.values()), key=lambda item: (item[0] - prev_code_length) / np.sum(item[3][-1]),
            )[0]
            if early_stop and code_length > code_lengths["sum"][-1]: break
            if np.sum(concepts_new[:, -1]) == 1: break  # the merged pattern is actually not common
            code_length, _, concepts, codebook = sorted(
                list(length_after_merge.values()), key=lambda item: (item[0] - prev_code_length) / np.sum(item[3][-1]),
            )[0]
            code_lengths["sum"].append(code_length)
        else:
            code_length, _, _, _ = sorted(
                list(length_after_merge.values()), key=lambda item: item[0],
            )[0]
            if early_stop and code_length > code_lengths["sum"][-1]: break
            code_length, _, concepts, codebook = sorted(
                list(length_after_merge.values()), key=lambda item: item[0],
            )[0]
            code_lengths["sum"].append(code_length)

        # update the saved data
        if objective == "codebook+total_length":
            code_lengths["words"].append(calculate_total_code_length(concepts, eval_val))
            code_lengths["codebook"].append(calculate_codebook_length(original_concepts, concepts, codebook, eval_val))
        elif objective.endswith("entropy+total_length"):
            code_lengths["words"].append(calculate_total_code_length(concepts, eval_val))
            code_lengths["entropy"].append(calculate_average_code_length(concepts, eval_val))
        elif objective.endswith("entropy+total_length-eff") or objective.endswith("entropy+total_length-eff-early"):
            code_lengths["words"].append(calculate_total_code_length(concepts, eval_val))
            code_lengths["entropy"].append(calculate_average_code_length(concepts, eval_val))
            decrease_per_dim = (code_lengths["sum"][-2] - code_lengths["sum"][-1]) / np.sum(codebook[-1])
            code_lengths["decrease-per-dim"].append(decrease_per_dim)
            code_lengths["merge-size"].append(np.sum(codebook[-1]))

    return codebook[original_concepts.shape[1]:], concepts, code_lengths


def visualize_coding_length(coding_length, save_path="test.png"):
    plt.figure(figsize=(8, 6))
    X = np.arange(1, len(coding_length) + 1)
    plt.plot(X, coding_length)
    plt.xlabel("iteration")
    plt.ylabel("code length")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close("all")


def calculate_edge_num(merged_patterns, aggregated_concepts):
    if aggregated_concepts.shape[0] == 0:
        return 0
    n_feature_dim = merged_patterns.shape[1]
    codebook = np.vstack([np.eye(n_feature_dim).astype(bool), merged_patterns])
    edge_num = np.any(aggregated_concepts, axis=1).sum()
    edge_num += aggregated_concepts.sum()
    is_code_used = np.any(aggregated_concepts, axis=0)
    is_code_used[:n_feature_dim] = np.logical_or(
        is_code_used[:n_feature_dim], np.any(merged_patterns[is_code_used[n_feature_dim:]], axis=0)
    )
    # edge_num += codebook[n_feature_dim:][is_code_used[n_feature_dim:]].sum()
    edge_num += merged_patterns[is_code_used[n_feature_dim:]].sum()
    return edge_num

def calculate_node_num(merged_patterns, aggregated_concepts):
    if aggregated_concepts.shape[0] == 0:
        return 0
    n_feature_dim = merged_patterns.shape[1]
    codebook = np.vstack([np.eye(n_feature_dim).astype(bool), merged_patterns])
    node_num = 1 + np.any(aggregated_concepts, axis=1).sum()
    is_code_used = np.any(aggregated_concepts, axis=0)
    is_code_used[:n_feature_dim] = np.logical_or(
        is_code_used[:n_feature_dim], np.any(merged_patterns[is_code_used[n_feature_dim:]], axis=0)
    )
    node_num += is_code_used.sum()
    return node_num

def calculate_tree_complexity():
    raise NotImplementedError


def visualize_CI_aggregated_pattern(attributes, concepts, eval_val, eval_type, judge_is_selected,
                                    aggregated_patterns, save_path="test.png"):

    n_feature_dim = len(attributes)

    aggregated_attributes = []
    for pattern in aggregated_patterns:
        description = []
        for i in range(pattern.shape[0]):
            if pattern[i]: description.append(attributes[i])
        description = "+".join(description)
        aggregated_attributes.append(description)

    # whether each concept contains some aggregated patterns
    aggregated_concepts = np.zeros(shape=(concepts.shape[0], len(aggregated_patterns))).astype(bool)
    for concept_id in range(len(concepts)):
        concept = concepts[concept_id]
        set_to_false_indice = np.zeros_like(concept).astype(bool)
        for pattern_id in range(len(aggregated_patterns)):
            pattern = aggregated_patterns[pattern_id]
            if is_A_subset_B(pattern, concept):
                aggregated_concepts[concept_id, pattern_id] = True
                set_to_false_indice = np.logical_or(set_to_false_indice, pattern)
        concepts[concept_id][set_to_false_indice] = False

    plt.figure(figsize=(20, 10))

    plt.subplot(2, 1, 2)
    ax_attribute = plt.gca()
    x = np.arange(len(concepts))
    y = np.arange(len(attributes) + len(aggregated_attributes))
    plt.xticks(x, [])
    plt.yticks(y, attributes + aggregated_attributes)
    plt.xlim(x.min() - 0.5, x.max() + 0.5)
    plt.ylim(y.min() - 0.5, y.max() + 0.5)
    plt.xlabel("concept")
    plt.ylabel("attribute")

    patch_colors = {
        True: {
            'pos': 'red',
            'neg': 'blue'
        },
        False: 'gray'
    }
    patch_width = 0.8
    patch_height = 0.9

    for concept_id in range(len(concepts)):
        concept = concepts[concept_id]
        for attribute_id in range(len(attributes)):
            is_selected = judge_is_selected(attributes[attribute_id], attribute_id, concept)
            if not is_selected:
                facecolor = patch_colors[is_selected]
            else:
                if eval_val[concept_id] > 0:
                    facecolor = patch_colors[is_selected]['pos']
                else:
                    facecolor = patch_colors[is_selected]['neg']
            rect = Rectangle(
                xy=(concept_id - patch_width / 2,
                    attribute_id - patch_height / 2),
                width=patch_width, height=patch_height,
                edgecolor=None,
                facecolor=facecolor,
                alpha=0.5
            )
            ax_attribute.add_patch(rect)

        aggregated_concept = aggregated_concepts[concept_id]
        for pattern_id in range(len(aggregated_attributes)):
            is_selected = aggregated_concept[pattern_id]
            if not is_selected:
                facecolor = patch_colors[is_selected]
            else:
                if eval_val[concept_id] > 0:
                    facecolor = patch_colors[is_selected]['pos']
                else:
                    facecolor = patch_colors[is_selected]['neg']
            rect = Rectangle(
                xy=(concept_id - patch_width / 2,
                    n_feature_dim + pattern_id - patch_height / 2),
                width=patch_width, height=patch_height,
                edgecolor=None,
                facecolor=facecolor,
                alpha=0.5
            )
            ax_attribute.add_patch(rect)



    plt.subplot(2, 1, 1, sharex=ax_attribute)
    plt.ylabel(eval_type)
    ax_eval = plt.gca()
    plt.setp(ax_eval.get_xticklabels(), visible=False)
    ax_eval.spines['right'].set_visible(False)
    ax_eval.spines['top'].set_visible(False)
    plt.plot(np.arange(len(concepts)), np.abs(eval_val))
    plt.hlines(y=0, xmin=0, xmax=len(concepts), linestyles='dotted', colors='red')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")


def plot_input_baseline(input, baseline, attributes, save_path):
    plt.figure(figsize=(8, 6))
    X = np.arange(input.shape[0])
    plt.xticks(np.arange(len(attributes)), attributes, rotation=90)
    plt.xlabel("feature dimension")
    plt.ylabel("value")
    plt.bar(X - 0.1, input, width=0.2, align="center", label="input")
    plt.bar(X + 0.1, baseline, width=0.2, align="center", label="baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close("all")


def plot_explain_ratio_tree_complexity(plot_dict: dict, save_path):
    for merge_time in plot_dict.keys():
        plot_dict[merge_time] = np.array(plot_dict[merge_time])
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.xlabel("explain ratio")
    plt.ylabel("#edge + #node")
    for merge_time in plot_dict.keys():
        plt.plot(plot_dict[merge_time][:, 0], plot_dict[merge_time][:, 1], label=f"{merge_time} merges")
    plt.legend()
    plt.subplot(132)
    plt.xlabel("explain ratio")
    plt.ylabel("#edge")
    for merge_time in plot_dict.keys():
        plt.plot(plot_dict[merge_time][:, 0], plot_dict[merge_time][:, 2], label=f"{merge_time} merges")
    plt.legend()
    plt.subplot(133)
    plt.xlabel("explain ratio")
    plt.ylabel("#node")
    for merge_time in plot_dict.keys():
        plt.plot(plot_dict[merge_time][:, 0], plot_dict[merge_time][:, 3], label=f"{merge_time} merges")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close("all")