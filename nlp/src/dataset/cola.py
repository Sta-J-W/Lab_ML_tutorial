import torch
import torchtext
import torchtext.legacy
from torchtext.legacy.data import TabularDataset, BucketIterator
import numpy as np
import os
import os.path as osp


def get_pad_to_min_len_fn(min_length):
    def pad_to_min_len(batch, vocab, min_length=min_length):
        pad_idx = vocab.stoi['<pad>']
        for idx, ex in enumerate(batch):
            if len(ex) < min_length:
                batch[idx] = ex + [pad_idx] * (min_length - len(ex))
        return batch
    return pad_to_min_len


def load_CoLA(args):
    data_root = osp.join(args.data_path, "CoLA")
    if args.gpu_id >= 0: device = torch.device("cuda")
    else: device = torch.device("cpu")
    if args.arch == "cnn":
        TEXT = torchtext.legacy.data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True,
                                           batch_first=True, postprocessing=get_pad_to_min_len_fn(5))
    else:
        TEXT = torchtext.legacy.data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True,
                                           batch_first=True)
    LABEL = torchtext.legacy.data.LabelField(dtype=torch.float)
    fields = [("id", None), ("label", LABEL), ("ori_label", None), ("text", TEXT)]
    train_set, test_set = TabularDataset.splits(
        path=data_root,
        format="tsv",
        train="train.tsv",
        test="dev.tsv",
        fields=fields
    )
    TEXT.build_vocab(train_set)
    LABEL.build_vocab(train_set)
    train_iterator, test_iterator = BucketIterator.splits(
        (train_set, test_set),
        batch_size=args.batch_size,
        sort_within_batch=True,
        sort_key=lambda x: len(x.text),
        device=device,
    )
    return train_set, test_set, train_iterator, test_iterator, TEXT, LABEL


if __name__ == '__main__':
    import argparse
    args = argparse.Namespace(
        data_path="/data1/limingjie/data/NLP",
        gpu_id=2,
        batch_size=100,
        arch="lstm"
    )
    _train_set, _test_set, _train_iterator, _test_iterator, _TEXT, _LABEL = load_CoLA(args)
    print(_LABEL.vocab.stoi)
    print("Train set:", len(_train_set))
    print("Test set:", len(_test_set))
    print("Sample sentence:", _train_set[0].text)
    print("Label of the sample sentence:", _LABEL.vocab.stoi[_train_set[0].label])

    for batch in _test_iterator:
        print(batch.text[0].shape)
        break

    print(len(list(_TEXT.vocab.stoi.values())))
    print(len(_TEXT.vocab))

