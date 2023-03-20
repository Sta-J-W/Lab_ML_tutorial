from .lstm import *
from .cnn import *


def load_nlp_model(args, TEXT, LABEL, model_path=None):
    if args.gpu_id >= 0:
        device = torch.device(f"cuda")
    else:
        device = torch.device("cpu")
    if args.dataset in ["CoLA", "SST-2"]:
        OUTPUT_DIM = 1
    else:
        raise NotImplementedError(f"[load_model.py (load_nlp_model)] Unknown dataset: {args.dataset}")

    if args.arch == "lstm2_uni":
        VOCAB_SIZE = len(TEXT.vocab)
        EMBEDDING_DIM = 100
        HIDDEN_DIM = 256
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
        model = lstm2_uni(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            pad_idx=PAD_IDX
        )
        if "pretrained_wordvec" in vars(args).keys() and args.pretrained_wordvec is not None:
            print("Use pretrained embeddings")
            pretrained_embeddings = TEXT.vocab.vectors
            model.embedding.weight.data.copy_(pretrained_embeddings)
            UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
            model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
            model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
        else:
            print(TEXT.vocab.vectors)
    elif args.arch == "cnn":
        VOCAB_SIZE = len(TEXT.vocab)
        EMBEDDING_DIM = 100
        N_FILTERS = 100
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
        model = cnn(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            n_filters=N_FILTERS,
            output_dim=OUTPUT_DIM,
            pad_idx=PAD_IDX
        )
        if "pretrained_wordvec" in vars(args).keys() and args.pretrained_wordvec is not None:
            print("Use pretrained embeddings")
            pretrained_embeddings = TEXT.vocab.vectors
            model.embedding.weight.data.copy_(pretrained_embeddings)
            UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
            model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
            model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
        else:
            print(TEXT.vocab.vectors)
    else:
        raise NotImplementedError(f"[load_model.py (load_nlp_model)] Unknown DNN: {args.arch}")

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        print(f"[load_model.py (load_nlp_model)] The pretrained model has been loaded from {model_path}.")
    else:
        print(f"[load_model.py (load_nlp_model)] Randomly initialize the model.")
    print(model)
    model = model.to(device)
    return model