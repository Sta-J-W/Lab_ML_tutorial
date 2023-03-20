import os
import torch
import torch.nn as nn
import numpy as np

# =========================
#   1. define the model
# =========================
net = nn.Sequential(
    nn.Linear(10, 100, bias=False),
    nn.ReLU(),
    nn.Linear(100, 100, bias=False),
    nn.ReLU(),
    nn.Linear(100, 100, bias=False),
    nn.ReLU(),
    nn.Linear(100, 10, bias=False)
).cuda()

# =============================
#   2. define input, baseline
# =============================
x = torch.randn(1, 10).cuda()
y = 2  # label
baseline = torch.zeros_like(x).cuda()
# define attribute names (for visualization use)
attributes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

# ==============================
#   3. calculate interaction
#     (you can do this by
#       either method below)
# ==============================
#  3.1 -- method 1
from interaction import ConditionedInteraction
calculator = ConditionedInteraction(mode="precise")
masks, harsanyi = calculator.calculate(
    model=net,
    input=x.squeeze(),
    baseline=baseline.squeeze(),
    selected_output_dim="gt-log-odds",
    gt=y
)  # masks: [2**n, n], each row indicates a subset S
#  3.2 -- method 2
from interaction import get_all_subset_rewards
from interaction.convert_matrix import get_reward2harsanyi_mat, get_harsanyi2reward_mat
masks, rewards = get_all_subset_rewards(
    model=net,
    input=x.squeeze(),
    baseline=baseline.squeeze(),
    selected_output_dim="gt-log-odds",
    gt=y
)
reward2harsanyi = get_reward2harsanyi_mat(all_masks=masks).cuda()
harsanyi_ = torch.matmul(reward2harsanyi, rewards)

#  verify results are the same
print(torch.max(torch.abs(harsanyi_ - harsanyi)))
#  verify efficiency axiom
v_N = torch.softmax(net(x), dim=1)[:, y].item()
v_N = np.log(v_N / (1 - v_N))
print(v_N, harsanyi.sum(), harsanyi_.sum())

# ==========================
#   4. visualize AOG
# ==========================
from tools.eval_ci import aggregate_pattern_iterative
from aog.aog_utils import construct_AOG_v1

harsanyi = harsanyi.cpu().numpy()
masks = masks.cpu().numpy()
x = x.squeeze().cpu().numpy()
baseline = baseline.squeeze().cpu().numpy()

#  For example, visualize top-25 interaction patterns
retained = np.argsort(-np.abs(harsanyi))[:25]

#  (1) merge common coalitions
merged_patterns, aggregated_concepts, coding_length = aggregate_pattern_iterative(
    concepts=masks[retained], eval_val=harsanyi[retained],
    max_iter=5, objective="10.0_entropy+total_length-eff-early"
)
aggregated_attributes = []
for pattern in merged_patterns:
    description = []
    for i in range(pattern.shape[0]):
        if pattern[i]: description.append(attributes[i])
    description = "+".join(description)
    aggregated_attributes.append(description)
single_features = np.vstack([np.eye(len(attributes)).astype(bool), merged_patterns])

#  (2) draw the AOG
arrow = {True: r"↑", False: r"↓"}
attributes_baselines = [rf"{attributes[i]}" + arrow[x[i] > baseline[i]] for i in range(len(attributes))]
aog = construct_AOG_v1(
    attributes=attributes_baselines,
    attributes_baselines=attributes_baselines,
    single_features=single_features,
    concepts=aggregated_concepts,
    eval_val=harsanyi[retained],
    remove_linebreak_in_coalition=len(merged_patterns) <= 2
)  # AOG 中不显示 I(emptyset)

figsize = (15, 7)
max_node = 9
os.makedirs("../tmp", exist_ok=True)
aog.visualize(
    save_path="../tmp/aog-demo.html", figsize=figsize,
    renderer="networkx", n_row_interaction=int(np.ceil(len(retained) / max_node)),
    title=f"output={harsanyi[retained].sum():.2f}"
)
aog.visualize(
    save_path="../tmp/aog.svg",
    renderer="networkx", n_row_interaction=int(np.ceil(len(retained) / max_node)),
    highlight_path=f"rank-1", title=f"output={harsanyi[retained].sum():.2f}", figsize=figsize
)

