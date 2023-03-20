from .cola import load_CoLA
from .sst2 import load_SST2


def load_nlp_dataset(args):
    if args.dataset == "CoLA":
        return load_CoLA(args)
    elif args.dataset == "SST-2":
        return load_SST2(args)
    else:
        raise NotImplementedError(f"[load_dataset.py (load_nlp_dataset)] Unknown dataset: {args.dataset}.")