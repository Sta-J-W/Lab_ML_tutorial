import json

import torch
from PIL import Image
import os
import os.path as osp
import numpy as np

import sys
sys.path.append(osp.join(osp.dirname(__file__), "../../.."))
from InteractionAOG_Image.src.tools.plot import plot_coalition, denormalize_image


def _get_majority(arr):
    """

    :param arr:
    :return:
    """
    elems = np.unique(arr)
    result = elems[0]
    count = np.sum((arr == result).astype(int))
    for elem in elems:
        if np.sum((arr == elem).astype(int)) > count:
            count = np.sum((arr == elem).astype(int))
            result = elem
    return result


def _downsample_label(label_hq, scale_factor):
    """

    :param label_hq:
    :param scale_factor:
    :return:
    """
    h_hq, w_hq = label_hq.shape
    label = np.zeros(shape=(h_hq // scale_factor, w_hq // scale_factor), dtype=np.uint8)
    h, w = label.shape
    for i in range(h):
        for j in range(w):
            label[i, j] = _get_majority(label_hq[i*scale_factor:(i+1)*scale_factor, j*scale_factor:(j+1)*scale_factor])
    return label


def generate_segments(label_path, target_shape):
    """

    :param label_path:
    :param target_shape:
    :return:
    """
    lbl = np.asarray(Image.open(label_path))
    h_hq, w_hq = lbl.shape
    h, w = target_shape
    assert h_hq // h == w_hq // w
    scale_factor = h_hq // h
    lbl = _downsample_label(label_hq=lbl, scale_factor=scale_factor)
    segment = []

    for seg_id in np.unique(lbl):
        if seg_id == 0:
            continue  # the background
        x, y = np.where(lbl == seg_id)
        segment.append((x * w + y).tolist())

    assert len(segment) == np.unique(lbl).max()
    return segment


def save_player_segment_normal():
    target_shape = (32, 32)
    manual_segment_root = "/data2/limingjie/InteractionAOG_Image/saved-manual-segments/simplemnist-segments/version-2"

    for class_id in os.listdir(manual_segment_root):
        if not osp.isdir(osp.join(manual_segment_root, class_id)):
            continue
        filenames = os.listdir(osp.join(manual_segment_root, class_id))
        for filename in filenames:
            if not osp.isdir(osp.join(manual_segment_root, class_id, filename)):
                continue
            label_path = osp.join(manual_segment_root, class_id, filename, "label.png")
            segment = generate_segments(label_path=label_path, target_shape=target_shape)

            with open(osp.join(manual_segment_root, class_id, f"{filename[:-3]}_players.json"), "w") as f:
                json.dump(segment, f, indent=4)

            plot_image = torch.load(osp.join(manual_segment_root, class_id, f"{filename[:-3]}_image.pth"))
            if plot_image.shape[-1] == 224:
                plot_image = denormalize_image(plot_image)
            plot_image = plot_image.squeeze(0).numpy()
            plot_coalition(
                image=plot_image, grid_width=1, coalition=segment,
                save_folder=osp.join(manual_segment_root, class_id),
                save_name=f"{filename[:-3]}_players",
                fontsize=15, linewidth=2, figsize=(4, 4)
            )



if __name__ == '__main__':
    save_player_segment_normal()
