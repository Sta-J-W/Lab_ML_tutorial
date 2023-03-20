import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_simple(nn.Module):

    def __init__(self, in_dim, hidd_dim, out_dim):
        super(MLP_simple, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidd_dim)
        self.fc2 = nn.Linear(hidd_dim, hidd_dim)
        self.fc3 = nn.Linear(hidd_dim, out_dim)

    def forward(self, x):
        x = x.reshape(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_MLP_simple(args):
    model_args = args.pretrained_args

    model = MLP_simple(
        in_dim=model_args["in_dim"],
        hidd_dim=model_args["hidd_dim"],
        out_dim=model_args["out_dim"]
    ).to(args.device)

    model.load_state_dict(torch.load(
        "dataset-{}-model-{}-epoch{}-seed{}-bs{}-logspace-{}-lr{}.pt".format(
            args.dataset, args.arch, model_args["epoch"], args.model_seed, args.batch_size, args.logspace, args.train_lr
        )
    ))
    print(model_args)


# ===========================
#   wrapper
# ===========================
def mlp(in_dim, hidd_dim, out_dim):
    return MLP_simple(in_dim, hidd_dim, out_dim)