import os

# import ML libs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .utils import makedirs
import os
import os.path as osp



def plot_curves(save_folder, res_dict):
    """plot curves for each key in dictionary.

    Args:
        args (dict): set containing all program arguments
        res_dict (dict): dictionary containing pairs of key and val(list)
    """
    for key in res_dict.keys():   
        # define the path
        path = os.path.join(save_folder, "{}-curve.png".format(key))
        # plot the fig
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(res_dict[key])) + 1, res_dict[key])
        ax.set(xlabel = 'epoch', ylabel = key, 
            title = '{}\'s curve'.format(key))
        ax.grid()
        fig.savefig(path)
        plt.close()


def plot_simple_line_chart(data, xlabel, ylabel, title, save_folder, save_name, X=None):
    makedirs(save_folder)
    plt.figure(figsize=(8, 6))
    plt.title(title)
    if X is None: X = np.arange(len(data))
    plt.plot(X, data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(osp.join(save_folder, f"{save_name}.png"), dpi=200)
    plt.close("all")


def compare_simple_line_chart(data_series, xlabel, ylabel, legends, title, save_folder, save_name, X=None):
    makedirs(save_folder)
    plt.figure(figsize=(8, 6))
    plt.title(title)
    for data, legend in zip(data_series, legends):
        if X is None: X = np.arange(len(data))
        plt.plot(X, data, label=legend)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(osp.join(save_folder, f"{save_name}.png"), dpi=200)
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

