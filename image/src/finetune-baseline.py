import json
import os
import os.path as osp
from pprint import pprint
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from datasets.get_dataset import get_dataset
import models.image_tiny as models
from harsanyi import AndHarsanyi, AndBaselineSparsifier
from harsanyi.interaction_utils import flatten, get_mask_input_func_image
from harsanyi.plot import visualize_pattern_interaction
from baseline_values import get_baseline_value
from tools.train import eval_model
from tools.plot import plot_coalition, visualize_all_interaction_descending, denormalize_image
from set_exp import setup_finetune_baseline


def parse_args():
    parser = argparse.ArgumentParser(description="evaluate the iou (on image datasets)")
    parser.add_argument('--data-root', default='/data2/limingjie/data', type=str,
                        help="root folder for dataset.")
    parser.add_argument("--model-args", type=str, default=None,
                        help="hyper-parameters for the pre-trained model")
    parser.add_argument("--segment-version", type=str, default=None,
                        help="the version of the manual segmentation for players")
    parser.add_argument('--gpu-id', default=0, type=int, help="set the device.")
    parser.add_argument("--model-root", default="../saved-models", type=str,
                        help='the root folder that stores the pre-trained model')
    parser.add_argument("--save-root", default="../saved-results/finetune-baseline", type=str,
                        help='the root folder to save results')
    parser.add_argument("--manual-segment-root", default="../saved-manual-segments", type=str,
                        help="the root folder for storing the segmentation info")

    parser.add_argument("--selected-dim", type=str, default="gt-log-odds",
                        help="use which dimension to compute interactions")

    # settings for finetuning baseline values
    parser.add_argument("--baseline-init", type=str, default="zero",
                        help="configuration of the baseline value")
    parser.add_argument("--baseline-lb", type=float, default=0.0,
                        help="lower bound of baseline value")
    parser.add_argument("--baseline-ub", type=float, default=0.1,
                        help="upper bound of baseline value")
    parser.add_argument("--finetune-lr", type=float, default=0.1,
                        help="learning rate of the baseline value")
    parser.add_argument("--finetune-max-iter", type=int, default=50,
                        help="number of iterations to finetune baseline value")
    parser.add_argument("--calc-bs", type=int, default=32,
                        help="the batch size when computing interactions")

    args = parser.parse_args()
    setup_finetune_baseline(args)
    return args


def get_model(args):
    model = models.__dict__[args.arch](**args.model_kwargs)
    model = model.to(args.gpu_id)
    model.load_state_dict(torch.load(osp.join(args.model_root, args.model_args, "model.pt"),
                                     map_location=torch.device(f"cuda:{args.gpu_id}")))
    model.eval()
    return model


def _get_denormalize_fn(dataset):
    if dataset in ["simplemnist", "simpleisthree"]:
        return lambda x: x
    elif dataset.startswith("celeba_"):
        return lambda x: denormalize_image(x)
    else:
        raise NotImplementedError


def _get_batch(folder):
    """
    get the input image, label, and player segments from the folder
      - in this folder, make sure all samples are with the same class
    :param folder: str, folder
    :return:
    """
    data_batch = {}
    sample_names = filter(lambda x: x.endswith("label.pth"), os.listdir(folder))
    sample_names = [sample_name[:-10] for sample_name in sample_names]
    sample_names = sorted(sample_names)
    for sample_name in sample_names:
        data_batch[sample_name] = {}
        data_batch[sample_name]["image"] = osp.join(folder, f"{sample_name}_image.pth")
        data_batch[sample_name]["label"] = osp.join(folder, f"{sample_name}_label.pth")
        data_batch[sample_name]["players"] = osp.join(folder, f"{sample_name}_players.json")
    return data_batch


def _visualize_players(plot_image, all_players, save_folder):
    plot_coalition(
        image=plot_image, grid_width=1, coalition=all_players,
        save_folder=save_folder, save_name="all_players",
        fontsize=15, linewidth=2, figsize=(4, 4)
    )
    for player_id, player in enumerate(all_players):
        plot_coalition(
            image=plot_image, grid_width=1, coalition=[flatten([player])],
            save_folder=osp.join(save_folder, "vis"), save_name=f"player_{player_id}",
            title=f"player: {player_id}", fontsize=15, linewidth=2, figsize=(4, 4)
        )


def _visualize_interaction(masks, interaction, plot_pattern_num, save_folder, save_name, attributes=None, title=""):
    n_dim = masks.shape[1]
    if attributes is None:
        attributes = [r"$x_{" + str(i) + r"}$" for i in range(n_dim)]
    if isinstance(interaction, torch.Tensor):
        interaction = interaction.clone().detach().cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.clone().detach().cpu().numpy()
    strength_order = np.argsort(-np.abs(interaction))
    visualize_pattern_interaction(
        coalition_masks=masks[strength_order[:plot_pattern_num]],
        interactions=interaction[strength_order[:plot_pattern_num]],
        attributes=attributes, title=title,
        save_path=osp.join(save_folder, f"plot_{save_name}.png"),
    )


def finetune_baseline_single(
        forward_func, selected_dim,
        image, baseline, label, calc_bs,
        all_players, finetune_kwargs,
        save_folder
):
    _, _, h, w = image.shape
    mask_input_fn = get_mask_input_func_image(grid_width=1)  # the grid width = 1
    foreground = list(flatten(all_players))
    indices = np.ones(h * w, dtype=bool)
    indices[foreground] = False
    background = np.arange(h * w)[indices].tolist()
    attributes = [r"$x_{" + str(i) + r"}$" for i in range(len(all_players))]

    # 1. calculate interaction
    calculator = AndHarsanyi(
        model=forward_func, selected_dim=selected_dim,
        x=image, baseline=baseline, y=label,
        all_players=all_players, background=background,
        mask_input_fn=mask_input_fn, calc_bs=calc_bs, verbose=0
    )
    with torch.no_grad():
        calculator.attribute()
        masks = calculator.get_masks()
        I_and_ = calculator.get_interaction()
        calculator.save(save_folder=osp.join(save_folder, "before_sparsify"))

    with open(osp.join(save_folder, "log.txt"), 'w') as f:  # 检查 efficiency 性质
        f.write("\n[Before Sparsifying]\n")
        f.write("sum of I^and:\n")
        f.write(f"\t{I_and_.sum()}\n")
        f.write("\n")

    # finetune baseline
    sparsifier = AndBaselineSparsifier(calculator=calculator, **finetune_kwargs)
    sparsifier.sparsify(verbose_folder=osp.join(save_folder, "sparsify_verbose"))
    with torch.no_grad():
        I_and = sparsifier.get_interaction()
        baseline_final = sparsifier.get_baseline()
        sparsifier.save(save_folder=osp.join(save_folder, "after_sparsify"))

    torch.save(I_and, osp.join(save_folder, "I_and.pth"))
    torch.save(baseline_final, osp.join(save_folder, "baseline_final.pth"))
    with open(osp.join(save_folder, "log.txt"), 'a') as f:  # 检查 efficiency 性质
        f.write("\n[After Sparsifying]\n")
        f.write(f"\tSum of I^and: {torch.sum(I_and)}\n")

    # 2. visualize interaction
    _visualize_interaction(
        masks=masks, interaction=I_and, plot_pattern_num=40,
        attributes=attributes, title=f"[AND patterns] v(N)={I_and.sum().item():.4f} | label: {label}",
        save_folder=save_folder, save_name="and_interaction",
    )
    visualize_all_interaction_descending(
        interactions=I_and.cpu().numpy(), save_path=osp.join(save_folder, "and_interaction_descreasing.png")
    )
    return I_and


if __name__ == '__main__':
    args = parse_args()

    # =========================================
    #     initialize the model and dataset
    # =========================================
    model = get_model(args)
    dataset = get_dataset(args.data_root, args.dataset)
    train_loader, test_loader = dataset.get_dataloader(batch_size=args.batch_size)
    denormalize = _get_denormalize_fn(args.dataset)

    # =========================================
    #    validate the pre-trained model
    # =========================================
    test_eval_dict = eval_model(model, test_loader, task=args.task)
    print("test loss:", test_eval_dict)

    for class_id in sorted(os.listdir(args.manual_segment_root)):
        if not osp.isdir(osp.join(args.manual_segment_root, class_id)) or class_id == "prototypes":
            continue
        print(f"Class id: {class_id}")
        data_batch = _get_batch(osp.join(args.manual_segment_root, class_id))

        for sample_id in data_batch.keys():
            save_folder = osp.join(args.save_root, class_id, sample_id)
            os.makedirs(save_folder, exist_ok=True)

            image = torch.load(osp.join(data_batch[sample_id]["image"]))
            label = torch.load(osp.join(data_batch[sample_id]["label"]))
            torch.save(image, osp.join(save_folder, "image.pth"))
            torch.save(label, osp.join(save_folder, "label.pth"))
            image = image.to(args.gpu_id)
            label = label.item()

            with open(data_batch[sample_id]["players"], "r") as f:
                all_players = json.load(f)
            with open(osp.join(save_folder, "all_players.json"), "w") as f:
                json.dump(all_players, f)

            print("sample", sample_id, "| # of players", len(all_players))
            # visualize the players
            plot_image = denormalize(image.squeeze(0).clone().cpu().numpy())
            _visualize_players(plot_image, all_players, save_folder)

            # initialize baseline value and finetune
            baseline_init = get_baseline_value(image, baseline_config=args.baseline_init)
            baseline_init = baseline_init.to(args.gpu_id)

            finetune_kwargs = {"loss": "l1", "baseline_min": args.baseline_lb, "baseline_max": args.baseline_ub,
                               "baseline_lr": args.finetune_lr, "niter": args.finetune_max_iter}
            I_and = finetune_baseline_single(
                forward_func=model, selected_dim=args.selected_dim,
                image=image, baseline=baseline_init, label=label, calc_bs=args.calc_bs,
                all_players=all_players, finetune_kwargs=finetune_kwargs,
                save_folder=save_folder
            )