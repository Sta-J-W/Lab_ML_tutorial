import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch


def generate_colorbar(ax, cmap_name, x_range, loc, title=""):
    '''
    generate a (fake) colorbar in a matplotlib plot
    :param ax:
    :param cmap_name:
    :param x_range:
    :param loc:
    :param title:
    :return:
    '''
    length = x_range[1] - x_range[0] + 1
    bar_ax = ax.inset_axes(loc)
    bar_ax.set_title(title)
    dummy = np.vstack([np.linspace(0, 1, length)] * 2)
    bar_ax.imshow(dummy, aspect='auto', cmap=plt.get_cmap(cmap_name))
    bar_ax.set_yticks([])
    bar_ax.set_xticks(x_range)


def plot_loss(loss, save_path, title=""):
    plt.figure(figsize=(8, 6))
    X = np.arange(1, len(loss) + 1)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.plot(X, loss)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close("all")


def plot_ci_trajectory(ci_trajectory, save_path, title=""):
    plt.figure(figsize=(8, 6))
    X = np.arange(1, len(ci_trajectory) + 1)
    plt.hlines(0, 0, X.shape[0], linestyles="dotted", colors="red")
    plt.xlabel("iteration")
    plt.ylabel(r"$I(S)$")
    plt.plot(X, ci_trajectory)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close("all")


def plot_coef_distribution(coefs, save_path, title=""):
    plt.figure(figsize=(8, 6))
    plt.hist(coefs.cpu().numpy(), bins=300)
    plt.xlabel("coefficient")
    plt.ylabel("count")
    plt.vlines(0, 0, 70, linestyles="dotted", colors="red")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close("all")



def plot_CI_mean(CI_mean, save_path, order_cfg="first", title=""):
    if not isinstance(CI_mean, list):
        CI_mean = [CI_mean]

    order_first = np.argsort(-CI_mean[0])

    plt.figure(figsize=(8, 6))
    plt.title(title)

    cmap_name = 'viridis'
    colors = cm.get_cmap(name=cmap_name, lut=len(CI_mean))
    colors = colors(np.arange(len(CI_mean)))

    label = None
    for i, item in enumerate(CI_mean):
        X = np.arange(1, item.shape[0] + 1)
        plt.hlines(0, 0, X.shape[0], linestyles="dotted", colors="red")
        label = f"iter {i+1}" if len(CI_mean) > 1 else None
        if order_cfg == "descending":
            plt.plot(X, item[np.argsort(-item)], label=label, color=colors[i])
        elif order_cfg == "first":
            plt.plot(X, item[order_first], label=label, color=colors[i])
        else:
            raise NotImplementedError(f"Unrecognized order configuration {order_cfg}.")
        plt.xlabel("patterns (with I(S) descending)")
        plt.ylabel("I(S)")
    # if label is not None: plt.legend()
    plt.tight_layout()
    ax = plt.gca()
    generate_colorbar(
        ax, cmap_name,
        x_range=(0, len(CI_mean) - 1),
        loc=[0.58, 0.9, 0.4, 0.03],
        title="iteration"
    )
    plt.savefig(save_path, dpi=200)
    plt.close("all")


def plot_baseline_values(baseline_list, save_path, title=""):
    baselines = torch.stack(baseline_list).detach().cpu().numpy()
    X = np.arange(1, baselines.shape[0] + 1)
    plt.figure(figsize=(8, 6))
    for i in range(baselines.shape[1]):
        plt.plot(X, baselines[:, i], label=f"feature-{i}")
        plt.xlabel("iteration")
        plt.ylabel("baseline value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close("all")


def plot_input_baseline(input, baseline, save_path):
    plt.figure(figsize=(8, 6))
    X = np.arange(input.shape[0])
    plt.xlabel("feature dimension")
    plt.ylabel("value")
    plt.bar(X - 0.1, input.clone().detach().cpu().numpy(), width=0.2, align="center", label="input")
    plt.bar(X + 0.1, baseline.clone().detach().cpu().numpy(), width=0.2, align="center", label="baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close("all")
