import torch


def unk_baseline(model, TEXT, device):
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    tensor = torch.LongTensor([UNK_IDX]).to(device)
    return model.get_emb(tensor).detach()


def pad_baseline(model, TEXT, device):
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    tensor = torch.LongTensor([PAD_IDX]).to(device)
    return model.get_emb(tensor).detach()