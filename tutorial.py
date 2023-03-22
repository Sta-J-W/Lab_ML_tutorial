### UNABLE TO TRAIN LOCALLY

#### Common imports ####
####======================================####
import torch
import torch.nn as nn

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#### Data transformation ####
####======================================####
import torchtext.transforms as T
from torch.hub import load_state_dict_from_url

padding_idx = 1
bos_idx = 0
eos_idx = 2
max_seq_len = 250
xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
xlmr_spm_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"

text_transform = T.Sequential(
    T.SentencePieceTokenizer(xlmr_spm_model_path), # 这一步将原句分词。输入：句子，输出：分词结果列表。
    T.VocabTransform(load_state_dict_from_url(xlmr_vocab_path)), # 这一步将分词结果转化为对应长度的词序。输入：词语列表，输出：对应长度向量。
    T.Truncate(max_seq_len),  # 截断
    T.AddToken(token=bos_idx, begin=True),
    T.AddToken(token=eos_idx, begin=False),
)
# Alternately we can also use transform shipped with pre-trained model that does all of the above out-of-the-box
# text_transform = XLMR_BASE_ENCODER.transform()

from torch.utils.data import DataLoader

#### Dataset####
####======================================####
from torchtext.datasets import SST2

batch_size = 16

train_datapipe = SST2(split="train")
dev_datapipe = SST2(split="dev")


# Transform the raw dataset using non-batched API (i.e apply transformation line by line)
def apply_transform(x):
    return text_transform(x[0]), x[1]


train_datapipe = train_datapipe.map(apply_transform)
train_datapipe = train_datapipe.batch(batch_size)
train_datapipe = train_datapipe.rows2columnar(["token_ids", "target"])
train_dataloader = DataLoader(train_datapipe, batch_size=None)

dev_datapipe = dev_datapipe.map(apply_transform)
dev_datapipe = dev_datapipe.batch(batch_size)
dev_datapipe = dev_datapipe.rows2columnar(["token_ids", "target"])
dev_dataloader = DataLoader(dev_datapipe, batch_size=None)

# Alternately we can also use batched API (i.e apply transformation on the whole batch)
def batch_transform(x):
    return {"token_ids": text_transform(x["text"]), "target": x["label"]}


train_datapipe = train_datapipe.batch(batch_size).rows2columnar(["text", "label"])
train_datapipe = train_datapipe.map(lambda x: batch_transform)
dev_datapipe = dev_datapipe.batch(batch_size).rows2columnar(["text", "label"])
dev_datapipe = dev_datapipe.map(lambda x: batch_transform)

#### Model Preparation ####
####======================================####
num_classes = 2
input_dim = 768
from torch.nn import Module
from torchtext.models import RobertaClassificationHead, XLMR_BASE_ENCODER

classifier_head = RobertaClassificationHead(num_classes=num_classes, input_dim=input_dim)
model = XLMR_BASE_ENCODER.get_model(head=classifier_head)
model.to(DEVICE)

# class MyClassifier(Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(10, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 16),
#             nn.ReLU(),
#             nn.Linear(16, 1)
#         )
#     def forward(self, x):
#         return torch.sigmoid(self.net(x)).squeeze(-1)

# model = MyClassifier().to(DEVICE)
#### Training Models ####
####========================================####
import torchtext.functional as F
from torch.optim import AdamW

learning_rate = 1e-5
optim = AdamW(model.parameters(), lr = learning_rate)
criteria = nn.CrossEntropyLoss()

def train_step(input, target):
    output = model(input)
    loss = criteria(output, target)
    optim.zero_grad()
    loss.backward()
    optim.step()

def eval_step(input, target):
    output = model(input)
    loss = criteria(output, target).item()
    # return float(loss), ((output > 0.5).int() == target.int()).type(torch.float).sum().item()
    return float(loss), (output.argmax(1) == target).type(torch.float).sum().item()

def evaluate():
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    counter = 0
    with torch.no_grad():
        for batch in dev_dataloader:
            # input = F.to_tensor(batch["token_ids"], padding_value=padding_idx).float().to(DEVICE)
            input = F.to_tensor(batch["token_ids"], padding_value=padding_idx).to(DEVICE)
            # target = torch.tensor(batch['target']).float().to(DEVICE)
            target = torch.tensor(batch['target']).to(DEVICE)
            loss, predictions = eval_step(input,target)
            total_loss += loss
            correct_predictions += predictions
            total_predictions += len(target)
            counter += 1
    return total_loss/counter, correct_predictions/total_predictions


#### Train ####
#### ====================================####

if __name__ == "__main__":
    num_epc = 1
    for e in range(num_epc):
        # for i, batch in enumerate(train_dataloader):
        #     if i % 50 == 0:
        #         print('.',end='')
        #     input = F.to_tensor(batch['token_ids'], padding_value=padding_idx).to(DEVICE)
        #     target = torch.tensor(batch['target']).to(DEVICE)
        #     train_step(input, target)
        loss, acc = evaluate()
        print("Epoch = [{}], loss = [{}], accuracy = [{}]".format(e, loss, acc))
    
    torch.save(model, 'nlp/nlp.pkl')



