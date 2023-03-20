import os

# import ML libs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .utils import makedirs
import os
import os.path as osp

# import internal libs
import seaborn as sns
from scipy.ndimage import gaussian_filter


def plot_prob(args, vals, labels, name, xlbl, ylbl):
    """plot hist for the array.

    Args:
        args (dict): set containing all program arguments
        val (np.array): (length, ) contain numbers. 
        labels (list): contain the label for each number.
        name (str): the name of the figure
        xlbl (str)
        ylbl (str)
    """
    fig, ax = plt.subplots()
    ax.bar(labels, vals)
    ax.set(xlabel = xlbl, ylabel = ylbl, title = name)
    # define the path & save the fig
    path = os.path.join(args.save_path, "{}.png".format(name))
    fig.savefig(path, dpi = 500)
    plt.close()


def plot_hist(args, arr, key, labels = None, if_density = True):
    """plot hist for the array.

    Args:
        args (dict): set containing all program arguments
        arr (np.array): (length, n) contain numbers. 
        labels (list): the list of label whose size corresponds to n.
        key (str): what the values stands for
        if_density (boolen): if use density.
    """
    fig, ax = plt.subplots()
    if labels == None:
        if if_density:
            ax.hist(arr, histtype='bar', density=True)
        else:
            ax.hist(arr, histtype='bar', density=False)
    else:
        if if_density:
            ax.hist(arr, histtype='bar', label=labels, density=True)
        else:
            ax.hist(arr, histtype='bar', label=labels, density=False)
        ax.legend(prop={'size': 10})
    ax.set(xlabel = key, title = '{}\'s distribution'.format(key))
    # define the path & save the fig
    path = os.path.join(args.save_path, "{}-hist.png".format(key))
    fig.savefig(path)
    plt.close()


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


def plot_curves_numpy(args, data, name):
    """plot curves for each key in dictionary.

    Args:
        args (dict): set containing all program arguments
        res_dict (dict): dictionary containing pairs of key and val(list)
    """
    # define the path
    path = os.path.join(args.save_path, "{}-curve.png".format(name))
    # plot the fig
    fig, ax = plt.subplots()
    ax.plot(np.arange(data.shape[0]) + 1, data)
    ax.set(xlabel='epoch', ylabel=name,
           title='{}\'s curve'.format(name))
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

